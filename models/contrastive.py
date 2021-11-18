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
        torch.nn.init.xavier_uniform_(self.embedding)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_probs(self, A_f: torch.Tensor) -> torch.Tensor:
        # TODO: fix this
        logits = A_f @ self.embedding.T # [b, d] @ [d, n] -> [b, n]
        return torch.sigmoid(logits)

    def contrastive_loss(labels: torch.Tensor, A_f: torch.Tensor) -> torch.Tensor:
        # We sum feature representations of multiple notes to make a chord feature representation.
        # TODO optionally add MLP layer to joint embedding.
        batch_size, num_labels = N_f.shape
        assert num_labels == self.num_labels
        N_f = labels @ self.embedding # [b,n] @ [n, d] -> [b, d]
        # Make sure the shapes match up.
        assert A_f.shape == N_f.shape
        # Normalize to create embeddings.
        # TODO(jxm) should I do bilinear interpolation here?
        A_e = tf.math.l2_normalize(A_f, axis=1)
        N_e = tf.math.l2_normalize(N_f, axis=1)
        logits = (A_e @ N_e.T) * torch.exp(self.temperature)
        # Symmetric loss function
        labels = torch.diag(torch.ones(batch_size)) # Identity matrix
        loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=0)
        loss_n = tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=1)
        return (loss_a + loss_n)/2