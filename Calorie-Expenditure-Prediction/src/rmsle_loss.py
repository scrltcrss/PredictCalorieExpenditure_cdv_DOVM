import torch
import torch.nn as nn


class RMSLELoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, min=0)
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
