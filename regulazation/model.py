import torch
import torch.nn as nn

class MLP(nn.Module):
    """Wine Classification MLP with 2 hidden layers"""
    def __init__(self, input_dim: int = 13, num_classes: int = 3, h1: int = 64, h2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
