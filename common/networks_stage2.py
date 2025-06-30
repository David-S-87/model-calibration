# networks_stage2.py

'''
Define neural networks for Stage 2: options + priors → posterior parameters.
Examples: PosteriorNet(), DeepSetPosteriorNet()
'''

import torch
import torch.nn as nn

class Stage2MLP(nn.Module):
    """
    Fully-connected MLP for Stage 2 amortized Bayesian update.
    Input: flattened option features concatenated with prior vector.
    Output: posterior parameter estimates.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # Second hidden layer
            nn.Linear(256, 256),
            nn.ReLU(),
            # Third hidden layer
            nn.Linear(256, 128),
            nn.ReLU(),
            # Output layer: predicts output_dim parameters
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: Tensor of shape (batch_size, input_dim)
        :return: Tensor of shape (batch_size, output_dim)
        """
        return self.net(x)


class Stage2DeepSets(nn.Module):
    """
    DeepSets architecture for Stage 2.
    Processes a set of M option feature vectors (each of dim feat_dim)
    and a prior vector, producing posterior parameters.
    """
    def __init__(self, feat_dim: int, num_options: int, prior_dim: int, output_dim: int):
        super().__init__()
        # φ: per-option embedding MLP
        self.phi = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # ρ: post-aggregation MLP
        # Input dim = aggregated φ output (64) + prior vector dim
        self.rho = nn.Sequential(
            nn.Linear(64 + prior_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, option_feats: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param option_feats: Tensor of shape (batch_size, num_options, feat_dim)
        :param prior:         Tensor of shape (batch_size, prior_dim)
        :return:              Tensor of shape (batch_size, output_dim)
        """
        batch_size, num_options, feat_dim = option_feats.shape
        # Flatten to (batch_size * num_options, feat_dim) for φ
        flattened = option_feats.view(-1, feat_dim)
        # Apply φ to each option
        phi_out = self.phi(flattened)                  # shape: (B * M, 64)
        # Reshape back to (batch_size, num_options, 64)
        phi_out = phi_out.view(batch_size, num_options, -1)
        # Sum over options to get permutation-invariant aggregate: (batch_size, 64)
        agg = phi_out.sum(dim=1)
        # Concatenate aggregated embedding with prior: (batch_size, 64 + prior_dim)
        combined = torch.cat([agg, prior], dim=1)
        # Apply ρ to get final parameter predictions
        return self.rho(combined)


def get_stage2_network(
    input_dim: int,
    output_dim: int,
    arch: str = "mlp",
    **kwargs
) -> nn.Module:
    """
    Factory function to construct a Stage 2 network.

    :param input_dim:  Dimension of the flat input (only for 'mlp' arch).
    :param output_dim: Number of parameters to predict.
    :param arch:       'mlp' or 'deepset'.
    :param kwargs:     If arch == 'deepset', requires:
                        - feat_dim:    int, dimension of each option feature vector
                        - num_options: int, number of options per set
                        - prior_dim:   int, dimension of prior vector
    :return:           An instance of nn.Module.
    """
    arch = arch.lower()
    if arch == "mlp":
        return Stage2MLP(input_dim, output_dim)
    elif arch == "deepset":
        return Stage2DeepSets(
            feat_dim=kwargs["feat_dim"],
            num_options=kwargs["num_options"],
            prior_dim=kwargs["prior_dim"],
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose 'mlp' or 'deepset'.")
