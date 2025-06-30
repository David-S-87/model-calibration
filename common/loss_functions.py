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

def stage2_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard mean squared error loss for Stage 2 posterior regression.
    """
    return F.mse_loss(predicted, target)

def stage2_mse_with_prior_reg(
    predicted: torch.Tensor,
    target: torch.Tensor,
    prior: torch.Tensor,
    alpha: float = 0.1
) -> torch.Tensor:
    """
    MSE loss with an added regularization term that penalizes deviation
    from the prior estimates.

    Loss = MSE(predicted, target) + alpha * MSE(predicted, prior)

    Args:
        predicted: Tensor of shape (B, D) — predicted posteriors.
        target:    Tensor of shape (B, D) — ground truth parameters.
        prior:     Tensor of shape (B, D) — prior estimates from Stage 1.
        alpha:     Weight of the prior-regularization term.
    """
    mse_term = F.mse_loss(predicted, target)
    reg_term = F.mse_loss(predicted, prior)
    return mse_term + alpha * reg_term
