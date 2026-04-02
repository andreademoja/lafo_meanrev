"""
CNN Filter for LAFO Mean-Reversion Strategy
Sections 2.1-2.3: Convolutional filtering for fair value estimation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import from same package directory
from lafo import lafo_loss

# Alias for consistency with other modules
class CNNFilter(nn.Module):
    """
    CNN filter for fair value estimation.
    Alias for LAFOCNN for API consistency.
    """
    def __init__(self, num_channels: int = 48, kernel_size: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(1, num_channels, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(num_channels, 1, kernel_size=1)

    def forward(self, x):
        # x must be exactly 3D: (batch, channels, length)
        if x.dim() == 4:
            x = x.squeeze(-1)  # remove extra dimension if present
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x.squeeze(1)

    def fit(self, y: np.ndarray, K: int = 20, num_epochs: int = 100, lr: float = 0.008):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Create tensor: always [1, 1, T]
        y_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            hat_y = self.forward(y_tensor)
            hat_y_np = hat_y.detach().cpu().numpy().squeeze()

            loss_val = lafo_loss(y, hat_y_np, K)
            loss = torch.tensor(loss_val, dtype=torch.float32, requires_grad=True)

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch [{epoch+1}/{num_epochs}]  LAFO Loss: {loss.item():.6f}")

        print("CNN training completed.")


# Main CNN class (alias for consistency)
class LAFOCNN(CNNFilter):
    """
    LAFO CNN filter with LAFO loss optimization.
    This is an alias for CNNFilter for backward compatibility.
    """
    pass
