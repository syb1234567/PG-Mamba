import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def soft_dice_loss(input: Tensor, target: Tensor, epsilon: float = 1e-6):


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

    def __init__(self, weight_dice=0.5, weight_mse=0.5, epsilon=1e-6, **kwargs):
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


        if predict.min() < 0 or predict.max() > 1:
            pred_prob = torch.sigmoid(predict)
        else:
            pred_prob = predict

        loss_dice = soft_dice_loss(pred_prob, target, self.epsilon)


        loss_mse = F.mse_loss(pred_prob, target)


        total_loss = (self.weight_dice * loss_dice) + (self.weight_mse * loss_mse)
        
        return total_loss


class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, predict, target):
        if predict.dim() == 4 and predict.size(1) == 1:
            predict = predict.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
            
        if predict.min() < 0 or predict.max() > 1:
            pred = torch.sigmoid(predict)
        else:
            pred = predict
        return soft_dice_loss(pred, target, self.epsilon)

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def forward(self, predict, target):
        if predict.dim() == 4 and predict.size(1) == 1:
            predict = predict.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        if predict.min() < 0 or predict.max() > 1:
            pred = torch.sigmoid(predict)
        else:
            pred = predict
            
        if target.dtype != torch.float32:
            target = target.float()
            
        return F.mse_loss(pred, target)

DiceLoss = SoftDiceLoss