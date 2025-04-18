# models/solar_time_model_tiny.py
import torch.nn as nn
class TinyMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1     = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32)
        )
        self.dropout = nn.Dropout(0.2)          
        self.fc2     = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.out     = nn.Linear(16, 1)         

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.out(x)                      
