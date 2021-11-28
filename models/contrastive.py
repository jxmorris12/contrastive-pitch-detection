import numpy as np
import torch
from torch import nn

class ContrastiveModel(nn.Module):
    embedding: nn.Embedding
    model: nn.Module
    batch_size: int
    def __init__(self, model: nn.Module, min_midi: int, max_midi: int, output_dim: int):
        super().__init__()
        # From CLIP: "The learnable temperature parameter
        # τ was initialized to the equivalent of 0.07 from (Wu et al.,
        # 2018) and clipped to prevent scaling the logits by more
        # than 100 which we found necessary to prevent training instability."
        self.temperature = 1.0 # TODO(jxm): consider making temperature learnable like in CLIP
        self.num_labels = (max_midi - min_midi + 1) # typically 88 (num notes on a piano)
        self.embedding = nn.Embedding(
            self.num_labels, output_dim
        )
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns representation of an audio input."""
        return self.model(x)

    def encode_note_labels(self, labels: torch.Tensor) -> torch.Tensor:
         return labels @ self.embedding.weight #  [b,n] @ [n, d] -> [b, d]

    def get_probs(self, audio_embeddings: torch.Tensor, n_steps=20, epsilon = 0.002, lr=10.0) -> torch.Tensor:
        """Returns note-wise probabilities for an audio input.
        
        Works by doing gradient descent to find the label to maximize the similarity
            between label_embeddings and audio_embeddings.
        """
        assert torch.is_grad_enabled(), "ContrastiveModel needs gradients to find most likely note labels"

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
            torch.clamp(labels, min=0.0, max=1.0)

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
            logits (float torch.Tensor):
        """
        # We sum embeddings of multiple notes to make a chord embedding.
        # TODO optionally add MLP layer to joint embedding.
        # breakpoint()
        batch_size, num_notes = note_labels.shape
        assert num_notes == self.num_labels
        chord_embeddings = self.encode_note_labels(note_labels)
        # Make sure the shapes match up.
        assert chord_embeddings.shape == audio_embeddings.shape
        # Normalize to create embeddings.
        # TODO(jxm) should I do bilinear interpolation here?
        normalized_audio_embeddings = audio_embeddings / torch.norm(audio_embeddings, dim=1, keepdim=True)
        normalized_chord_embeddings = chord_embeddings / torch.norm(chord_embeddings, dim=1, keepdim=True)
        logits = (normalized_audio_embeddings @ normalized_chord_embeddings.T) * np.exp(self.temperature)
        # Symmetric loss function
        labels = torch.diag(torch.ones(batch_size)).to(logits.device) # Identity matrix
        loss_a = torch.nn.functional.binary_cross_entropy_with_logits(labels, logits)
        loss_n = torch.nn.functional.binary_cross_entropy_with_logits(labels, logits.T)

        loss = (loss_a + loss_n)/2
        return loss, logits