# networks_stage1.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from config_template import get_param_names


class Stage1Network(nn.Module):
    """
    Multi-layer perceptron for Stage 1: summary stats -> parameter estimates.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(Stage1Network, self).__init__()
        # Define a simple 3-layer MLP
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),  # first hidden layer
            nn.ReLU(),                   # activation
            nn.Linear(128, 128),         # second hidden layer
            nn.ReLU(),                   # activation
            nn.Linear(128, output_dim)   # output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute parameter predictions.
        Input:
            x: tensor of shape (batch_size, input_dim)
        Output:
            tensor of shape (batch_size, output_dim)
        """
        return self.model(x)


def get_stage1_network(input_dim: int, output_dim: int) -> Stage1Network:
    """
    Factory to create a Stage1Network with given dimensions.
    """
    return Stage1Network(input_dim, output_dim)


def train_stage1_model(
    model_type: str,
    dataset_path: str,
    save_path: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = None
) -> None:
    """
    Train the Stage 1 network for the given model type using synthetic dataset.

    Saves the trained weights to save_path.
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load synthetic dataset
    data = np.load(dataset_path)
    X = data['summary_stats']    # shape: (N, d_stats)
    y = data['true_params']      # shape: (N, d_params)
    param_names = data['param_names']  # not used directly here

    # Convert to torch tensors
    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create network
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    net = get_stage1_network(input_dim, output_dim).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Training loop
    net.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()          # reset gradients
            preds = net(batch_X)           # forward pass
            loss = criterion(preds, batch_y)  # compute MSE loss
            loss.backward()                # backpropagate
            optimizer.step()               # update weights

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save model weights
    torch.save(net.state_dict(), save_path)
    print(f"Saved Stage1 model to {save_path}")
