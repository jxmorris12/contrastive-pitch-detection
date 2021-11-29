import numpy as np
import torch
from torch import nn

class ContrastiveModel(nn.Module):
    """A CLIP-inspired contrastive learning model for jointly learning chords and audio."""
    embedding: nn.Embedding
    model: nn.Module
    batch_size: int
    def __init__(self, model: nn.Module, min_midi: int, max_midi: int, output_dim: int):
        super().__init__()
        # From CLIP: "The learnable temperature parameter
        # τ was initialized to the equivalent of 0.07 from (Wu et al.,
        # 2018) and clipped to prevent scaling the logits by more
        # than 100 which we found necessary to prevent training instability."
        self.temperature = torch.nn.parameter.Parameter(
            torch.tensor(0.07, dtype=torch.float32), requires_grad=True)
        self.num_labels = (max_midi - min_midi + 1) # typically 88 (num notes on a piano)
        # TODO(jxm): Consider a different dimensionality for embedding and projection.
        embedding_dim = output_dim
        self.embedding = nn.Embedding(
            self.num_labels, embedding_dim
        )
        self.embedding_proj = nn.Linear(
            in_features=embedding_dim, out_features=output_dim
        )
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns representation of an audio input."""
        return self.model(x)

    def encode_note_labels(self, labels: torch.Tensor) -> torch.Tensor:
        # TODO(jxm): Consider concatenating + padding instead of summing note embddings.
        joint_embedding = labels @ self.embedding.weight #  [b,n] @ [n, d] -> [b, d]
        return self.embedding_proj(joint_embedding)

    def get_probs(self, audio_embeddings: torch.Tensor, n_steps=20, epsilon = 0.002, lr=10.0) -> torch.Tensor:
        """Returns note-wise probabilities for an audio input.
        
        Works by doing gradient descent to find the label to maximize the similarity
            between label_embeddings and audio_embeddings.

        Args:
            audio_embeddings (torch.Tensor): The audio embeddings for which we want labels.
            n_steps (int): Maximum number of steps before convergence. 
            epsilon (float): Minimum amount of change in cosine similarity between audio and
                note embeddings between steps. If change drops below this amount, stops.
            lr (float): Learning rate for SGD.
        """
        assert torch.is_grad_enabled(), "ContrastiveModel needs gradients enabled"

        if audio_embeddings.requires_grad:
            # Prevent PyTorch from tracking the computational graph for the audio
            # embeddings. We only want to keep gradients wrt `labels` below.
            audio_embeddings = audio_embeddings.detach()

        labels = torch.rand(
            (len(audio_embeddings), self.num_labels),
            dtype=torch.float32, requires_grad=True,
            device=audio_embeddings.device
        )
        optimizer = torch.optim.SGD([labels], lr=lr, momentum=0.9)
        cos_sim = torch.nn.CosineSimilarity(dim=1)
        last_loss = 1.0
        for _ in range(n_steps):
            label_embeddings = self.encode_note_labels(labels)
            loss = torch.mean(1 - cos_sim(audio_embeddings, label_embeddings))
            # print(f'Similarity at step {_}: {(1-loss).item():.3f}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            labels = torch.clamp(labels, min=0.0, max=1.0).detach().requires_grad_(True)

            if torch.abs(loss - last_loss) < epsilon:
                break
            last_loss = loss        
            
        return labels.detach()

    def contrastive_loss(self, audio_embeddings: torch.Tensor, note_labels: torch.Tensor) -> torch.Tensor:
        """Computes the contrastive loss of CLIP.

        Arguments:
            audio_embeddings (float torch.Tensor): model outputs for audio inputs of shape [batch_size, self.embedding.output_dim]
                example shape: [32, 256]
            note_labels (torch.tensor): one-hot chord labels of shape [batch_size, self.num_labels]
                example shape: [32, 88]
        
        Returns:
            loss (float torch.Tensor): scalar output, loss across both dimensions
            logits (float torch.Tensor): audio<->chord logits of shape (batch_size, batch_size)
        """
        # We sum embeddings of multiple notes to make a chord embedding.
        batch_size, num_notes = note_labels.shape
        assert num_notes == self.num_labels
        chord_embeddings = self.encode_note_labels(note_labels)
        assert chord_embeddings.shape == audio_embeddings.shape
        # Normalize embeddings and compute logits.
        normalized_audio_embeddings = audio_embeddings / torch.norm(audio_embeddings, p=2, dim=1, keepdim=True)
        normalized_chord_embeddings = chord_embeddings / torch.norm(chord_embeddings, p=2, dim=1, keepdim=True)
        unscaled_audio_to_chord_sim = torch.matmul(normalized_audio_embeddings, normalized_chord_embeddings.T)
        audio_to_chord_sim = unscaled_audio_to_chord_sim * torch.exp(self.temperature)
        chord_to_audio_sim = audio_to_chord_sim.T
        # Compute labels when there may be duplicates.
        labels = (note_labels[:,None] == note_labels).all(2).type(torch.float32)
        labels = labels / labels.sum(1)
        # Compute loss across both axes.
        loss_a = torch.nn.functional.cross_entropy(audio_to_chord_sim, labels)
        loss_n = torch.nn.functional.cross_entropy(chord_to_audio_sim, labels.T)
        loss = (loss_a + loss_n)/2
        return loss, unscaled_audio_to_chord_sim