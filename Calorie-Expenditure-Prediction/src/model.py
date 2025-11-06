import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    # Some of the best models 05976
    def __init__(self, size=7) -> None:
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(size, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
