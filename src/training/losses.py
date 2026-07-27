import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    """
    CLIP-style symmetric contrastive loss.

    For a batch of N (satellite, phone) pairs, computes bidirectional
    cross-entropy: satellite→phone + phone→satellite.

    The temperature parameter controls the sharpness of the softmax
    distribution. It is learnable (log-parameterized, clipped to
    [0.01, 100]) following the original CLIP formulation.
    """

    def __init__(self, temperature_init: float = 0.07):
        super().__init__()
        # Learnable temperature, stored as log(temp) for numerical stability
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(1.0 / temperature_init))
        )

    @property
    def temperature(self) -> torch.Tensor:
        # Clamp to prevent logits from exceeding 100x scale
        return torch.clamp(1.0 / self.log_temperature.exp(), min=0.01, max=100.0)

    def forward(
        self, sat_emb: torch.Tensor, phone_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sat_emb:   [batch_size, embed_dim] — L2-normalized satellite embeddings
            phone_emb: [batch_size, embed_dim] — L2-normalized phone embeddings

        Returns:
            Scalar loss (average of satellite→phone and phone→satellite)
        """
        temp = self.temperature

        # NxN similarity matrix
        logits = sat_emb @ phone_emb.T / temp  # [N, N]

        # Diagonal = positive pairs (matching location)
        labels = torch.arange(len(logits), device=logits.device)

        # Bidirectional cross-entropy
        loss_s2p = F.cross_entropy(logits, labels)
        loss_p2s = F.cross_entropy(logits.T, labels)

        return (loss_s2p + loss_p2s) / 2


class DistillationLoss(nn.Module):
    """
    MSE loss between teacher and student embeddings.

    Used to transfer the teacher's (ResNet-50) embedding knowledge
    to the student (MobileNetV3) for mobile deployment.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, student_emb: torch.Tensor, teacher_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_emb:  [batch_size, embed_dim] — student embeddings
            teacher_emb:  [batch_size, embed_dim] — frozen teacher embeddings

        Returns:
            Scalar MSE loss
        """
        return F.mse_loss(student_emb, teacher_emb.detach())
