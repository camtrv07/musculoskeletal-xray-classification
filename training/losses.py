"""
Loss functions for MURA abnormality detection.
Handles class imbalance with weighted Binary Cross-Entropy.
"""

import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for handling class imbalance.

    Applies higher weight to positive (abnormal) class to account for
    the 38.5% positive vs 61.5% negative distribution in the dataset.
    """

    def __init__(self, pos_weight=1.597):
        """
        Initialize weighted BCE loss.

        Args:
            pos_weight (float): Weight for positive class. Default: 1.597
                Calculated as: num_negatives / num_positives = 0.615 / 0.385
        """
        super().__init__()

        self.pos_weight = torch.tensor([pos_weight])
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        print(f"Initialized WeightedBCELoss with pos_weight={pos_weight:.4f}")

    def forward(self, logits, targets):
        """
        Compute weighted BCE loss.

        Args:
            logits: [batch, 1] - Model predictions (before sigmoid)
            targets: [batch] - Ground truth labels (0 or 1)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Ensure pos_weight is on the same device
        if self.loss_fn.pos_weight.device != logits.device:
            self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(logits.device)

        # Squeeze logits to match targets shape: [batch]
        logits = logits.squeeze(1)

        # Compute loss
        loss = self.loss_fn(logits, targets)

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by down-weighting easy examples.

    Alternative to weighted BCE if the imbalance is more severe.
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Initialize Focal Loss.

        Args:
            alpha (float): Weighting factor for positive class. Default: 0.25
            gamma (float): Focusing parameter (0 = standard CE). Default: 2.0
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        print(f"Initialized FocalLoss with alpha={alpha}, gamma={gamma}")

    def forward(self, logits, targets):
        """
        Compute Focal Loss.

        Args:
            logits: [batch, 1] - Model predictions (before sigmoid)
            targets: [batch] - Ground truth labels (0 or 1)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Squeeze logits
        logits = logits.squeeze(1)

        # Compute BCE
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute pt (probability of correct class)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Compute alpha weight
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Final loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        return focal_loss.mean()
