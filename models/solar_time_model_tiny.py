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

        # Layer 1: Input -> Hidden (input_dim -> 32)
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 32),   # Input: (batch_size, input_dim), Output: (batch_size, 32)
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        # Layer 2: Hidden -> Hidden (32 -> 16)
        self.fc2 = nn.Sequential(
            nn.Linear(32, 16),           # Output: (batch_size, 16)
            nn.ReLU()
        )
        
        # Output layer: Hidden -> Output (8 -> 1)
        self.out = nn.Linear(16, 1)      # Output: (batch_size, 1) — for regression

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
