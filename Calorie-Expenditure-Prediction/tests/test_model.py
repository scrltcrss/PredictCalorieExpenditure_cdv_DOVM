import torch

from src.workout_dataset import WorkoutDataset, load_data
from src.model import NeuralNetwork


def test_load_data(tmp_path: str):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "id,Age,Height,Weight,Duration,Heart_Rate,Body_Temp,Sex,Calories\n"
        "1,25,175,70,30,120,37.0,male,300\n"
        "2,30,160,60,20,110,36.5,female,250\n"
    )
    
    x, y = load_data(str(csv_file))
    assert x.shape == (2, 7)
    assert y.shape == (2, 1)
    assert (x[:, -1] <= 1).all()


def test_workout_dataset_getitem(tmp_path: str):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "id,Age,Height,Weight,Duration,Heart_Rate,Body_Temp,Sex,Calories\n"
        "1,25,175,70,30,120,37.0,male,300\n"
    )
    dataset = WorkoutDataset(str(csv_file))
    x, y = dataset[0]
    assert x.shape[0] == 7
    assert y.shape == (1,)
    assert len(dataset) == 1


def test_model_forward():
    model = NeuralNetwork()
    x = torch.randn(4, 7)
    output = model(x)
    assert output.shape == (4, 1)
