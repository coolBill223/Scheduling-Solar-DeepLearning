import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, input_dim: int):
        """
        Parameters:
            input_dim (int): The number of features for each input sample.
                             In your case, this should be 24.
        """
        super(TinyMLP, self).__init__()

        # Layer 1: Input -> Hidden (24 -> 16)
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 16),   # Input: (batch_size, 24), Output: (batch_size, 16)
            nn.ReLU(),
            nn.Dropout(0.2)             # Prevent overfitting
        )

        # Layer 2: Hidden -> Hidden (16 -> 8)
        self.fc2 = nn.Sequential(
            nn.Linear(16, 8),           # Output: (batch_size, 8)
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Output layer: Hidden -> Output (8 -> 1)
        self.out = nn.Linear(8, 1)      # Output: (batch_size, 1) — for regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: Output tensor of shape (batch_size, 1)
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x
