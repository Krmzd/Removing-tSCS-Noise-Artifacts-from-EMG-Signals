import torch
import torch.nn as nn

# Huber Loss
def get_huber_loss(beta=0.1):
    return nn.SmoothL1Loss(beta=beta)

# Log-Cosh Loss
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # log(cosh(x)) can be simplified for numerical stability:
        # log((exp(x) + exp(-x))/2)
        err = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(err + 1e-12)))

# Weighted Loss 
class ScaledLoss(nn.Module):
    def __init__(self, base_loss_fn, scale=100.0):
        super().__init__()
        self.loss_fn = base_loss_fn
        self.scale = scale

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true) * self.scale