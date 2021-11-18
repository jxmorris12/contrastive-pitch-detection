import numpy as np
import torch
from torch import nn

class ContrastiveModel(nn.Module):
    embedding: nn.Embedding
    model: nn.Module
    batch_size: int
    def __init__(self, model: nn.Module, min_midi: int, max_midi: int, output_dim: int):
        super().__init__()
        self.temperature = 1.0 # TODO(jxm): consider making temperature learnable like in CLIP
        self.num_labels = (max_midi - min_midi + 1)
        self.embedding = nn.Embedding(
            self.num_labels, output_dim
        )
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_probs(self, A_f: torch.Tensor) -> torch.Tensor:
        # TODO: fix this
        logits = A_f @ self.embedding.weight.T # [b, d] @ [d, n] -> [b, n]
        return torch.sigmoid(logits)

    def contrastive_loss(self, notes: torch.Tensor, A_f: torch.Tensor) -> torch.Tensor:
        # We sum feature representations of multiple notes to make a chord feature representation.
        # TODO optionally add MLP layer to joint embedding.
        batch_size, num_notes = A_f.shape
        assert num_notes == self.num_labels
        N_f = notes @ self.embedding.weight.T # [b,n] @ [n, d] -> [b, d]
        # Make sure the shapes match up.
        assert A_f.shape == N_f.shape
        # Normalize to create embeddings.
        # TODO(jxm) should I do bilinear interpolation here?
        A_e = A_f / torch.norm(A_f, dim=1, keepdim=True)
        N_e = N_f / torch.norm(N_f, dim=1, keepdim=True)
        logits = (A_e @ N_e.T) * np.exp(self.temperature)
        # Symmetric loss function
        labels = torch.diag(torch.ones(batch_size)).to(logits.device) # Identity matrix
        loss_a = torch.nn.functional.binary_cross_entropy_with_logits(labels, logits)
        loss_n = torch.nn.functional.binary_cross_entropy_with_logits(labels.T, logits.T)
        return (loss_a + loss_n)/2