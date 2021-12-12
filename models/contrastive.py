import functools

import numpy as np
import torch
from torch import nn

class ContrastiveModel(nn.Module):
    """A CLIP-inspired contrastive learning model for jointly learning chords and audio."""
    embedding: nn.Embedding
    model: nn.Module
    batch_size: int
    def __init__(self, 
            model: nn.Module,
            min_midi: int, max_midi: int,
            output_dim: int, max_polyphony: int
        ):
        super().__init__()
        assert 1 <= max_polyphony <= 6
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

        self.embedding_proj = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim, out_features=embedding_dim*2
            ),
            nn.BatchNorm1d(embedding_dim*2),
            nn.ReLU(),
            nn.Linear(
                in_features=embedding_dim*2, out_features=output_dim
            )
        )
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.model = model
        self.max_polyphony = max_polyphony
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns representation of an audio input."""
        return self.model(x)

    def encode_note_labels(self, labels: torch.Tensor) -> torch.Tensor:
        # TODO(jxm): Consider concatenating + padding instead of summing note embddings.
        joint_embedding = labels @ self.embedding.weight #  [b,n] @ [n, d] -> [b, d]
        return self.embedding_proj(joint_embedding)

    def _get_probs_single_note(self, audio_embedding: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Returns label for a single audio waveform, found with beam search."""
        # TODO support beam width > 1?
        cos_sim = functools.partial(torch.nn.functional.cosine_similarity, dim=1)
        best_labels = torch.zeros((self.num_labels))
        zero_label_encoding = self.encode_note_labels(best_labels[None].to(self.device))
        best_overall_sim = cos_sim(audio_embedding.squeeze(), zero_label_encoding).item()
        for _ in range(1, self.max_polyphony+1):
            new_labels = best_labels.repeat((self.num_labels,1))
            new_notes = torch.eye(self.num_labels)
            new_labels = torch.maximum(new_notes, new_labels) # 88 tensors, each one has a new 1 at a different position
            label_encodings = self.encode_note_labels(new_labels.to(self.device))
            cos_sims = cos_sim(audio_embedding, label_encodings)
            best_idx = cos_sims.argmax()
            best_sim = cos_sims[best_idx].item()
            
            if best_sim - best_overall_sim > epsilon:
                best_overall_sim = best_sim
                best_labels = new_labels[best_idx]
            else:
                break
        return best_labels.to(self.device)

    def get_probs(self, audio_embeddings: torch.Tensor, n_steps=20, epsilon = 0.00) -> torch.Tensor:
        """Returns note-wise probabilities for an audio input.
        
        Works by doing beam search to find the label to maximize the similarity
            between label_embeddings and audio_embeddings.

        Args:
            audio_embeddings (torch.Tensor): The audio embeddings for which we want labels.
            epsilon (float): Minimum amount of change in cosine similarity between audio and
                note embeddings between steps. If change drops below this amount, stops.
        """
        
        return torch.stack([self._get_probs_single_note(e, epsilon) for e in audio_embeddings])

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
        loss_a = torch.nn.functional.binary_cross_entropy(
            torch.nn.functional.softmax(audio_to_chord_sim, 0), labels)
        loss_n = torch.nn.functional.binary_cross_entropy(
            torch.nn.functional.softmax(chord_to_audio_sim, 0), labels)
        loss = (loss_a + loss_n)/2
        return loss, (loss_a, loss_n, audio_to_chord_sim)