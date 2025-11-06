import torch
from torch.utils.data import Dataset
import pandas as pd


def load_data(csv_file: str) -> tuple[torch.Tensor, torch.Tensor]:
    data: pd.DataFrame = pd.read_csv(csv_file)
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0}).astype(float)

    features: pd.DataFrame = data.drop(columns=['Calories', 'id'])
    target: pd.DataFrame = data['Calories']

    x: torch.Tensor = torch.tensor(features.values, dtype=torch.float32)
    y: torch.Tensor = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)
    
    return x, y


class WorkoutDataset(Dataset):

    def __init__(self, csv_file: str, normalize: bool = True) -> None:
        x, y = load_data(csv_file=csv_file)
        self.x: torch.Tensor = x
        self.y: torch.Tensor = y

        if normalize:
            x_min, _ = x.min(dim=0, keepdim=True)
            x_max, _ = x.max(dim=0, keepdim=True)
            self.x = (x - x_min) / (x_max - x_min + 1e-8)  # avoid division by zero
        
    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
