"""
Loss functions for soft-label retinal vessel segmentation.

The model outputs probability maps directly.
All losses operate on probability predictions without
additional sigmoid activation.

Implemented losses:
- Soft Dice loss
- Combined Dice + MSE loss
- MSE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def soft_dice_loss(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """
    Compute soft Dice loss for probability maps.

    Args:
        input (Tensor): Predicted probability map.
        target (Tensor): Soft segmentation label.
        epsilon (float): Numerical stability term.

    Returns:
        Tensor: Soft Dice loss value.
    """

    if input.dim() == 4 and input.size(1) == 1:
        input = input.squeeze(1)

    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    batch_size = input.size(0)

    input_flat = input.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    intersection = 2 * (input_flat * target_flat).sum(dim=1)
    union = input_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (intersection + epsilon) / (union + epsilon)

    return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Soft Dice and MSE loss.

    This loss is designed for soft-label segmentation,
    where predictions and targets are continuous values.
    """

    def __init__(
        self,
        weight_dice=0.5,
        weight_mse=0.5,
        epsilon=1e-6,
        **kwargs,
    ):
        super(CombinedLoss, self).__init__()

        self.weight_dice = weight_dice
        self.weight_mse = weight_mse
        self.epsilon = epsilon

    def forward(self, predict, target):
        if predict.dim() == 4 and predict.size(1) == 1:
            predict = predict.squeeze(1)

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        if target.dtype != torch.float32:
            target = target.float()

        pred_prob = predict

        loss_dice = soft_dice_loss(
            pred_prob,
            target,
            self.epsilon,
        )

        loss_mse = F.mse_loss(
            pred_prob,
            target,
        )

        total_loss = (
            self.weight_dice * loss_dice
            + self.weight_mse * loss_mse
        )

        return total_loss


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss module.
    """

    def __init__(self, epsilon=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predict, target):

        if predict.dim() == 4 and predict.size(1) == 1:
            predict = predict.squeeze(1)

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        return soft_dice_loss(
            predict,
            target,
            self.epsilon,
        )


class MSELoss(nn.Module):
    """
    Mean squared error loss for probability maps.
    """

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predict, target):

        if predict.dim() == 4 and predict.size(1) == 1:
            predict = predict.squeeze(1)

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        if target.dtype != torch.float32:
            target = target.float()

        return F.mse_loss(
            predict,
            target,
        )


DiceLoss = SoftDiceLoss