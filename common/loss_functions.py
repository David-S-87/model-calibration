# loss_functions.py

import torch
import torch.nn.functional as F

def stage1_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard mean squared error loss for Stage 1 parameter regression.
    """
    return F.mse_loss(predicted, target)

def stage1_log_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE on log-transformed values to stabilize training across parameters
    that vary over orders of magnitude.
    """
    return F.mse_loss(torch.log1p(predicted), torch.log1p(target))

def stage1_weighted_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Weighted MSE loss where `weights` has shape (n_params,) and specifies
    the relative importance of each parameter.
    """
    return torch.mean(weights * (predicted - target) ** 2)
