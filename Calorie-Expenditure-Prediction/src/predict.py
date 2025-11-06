"""
Prediction script for generating competition-format submission files.
"""

from typing import Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from architectures import get_model


class CaloriePredictor:
    """
    Prediction utility for trained models with ensemble support.
    """
    
    def __init__(
        self,
        model_paths: List[str],
        architecture: str = 'simple',
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        **model_kwargs
    ) -> None:
        self.model_paths = model_paths
        self.architecture = architecture
        self.dropout = dropout
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_kwargs = model_kwargs
        self.models: List[nn.Module] = []
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all model checkpoints."""
        for path in self.model_paths:
            model = get_model(
                self.architecture,
                input_size=7,
                dropout=self.dropout,
                **self.model_kwargs
            )
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models from checkpoints")
    
    def prepare_test_data(
        self,
        test_path: str,
        train_path: Optional[str] = None
    ) -> tuple[torch.Tensor, pd.DataFrame]:
        """
        Load and prepare test data for prediction.
        
        Args:
            test_path: Path to test CSV
            train_path: Optional path to training CSV for normalization
        
        Returns:
            Tuple of (prepared features tensor, original test dataframe)
        """
        test_df = pd.read_csv(test_path)
        test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1}).astype('float32')
        
        features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        X_test = torch.tensor(test_df[features].values, dtype=torch.float32)
        
        if train_path:
            train_df = pd.read_csv(train_path)
            train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1}).astype('float32')
            X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
            
            means = X_train.mean(dim=0, keepdim=True)
            stds = X_train.std(dim=0, keepdim=True)
            stds = torch.where(stds == 0, torch.ones_like(stds), stds)
            
            X_test = (X_test - means) / stds
        
        return X_test, test_df
    
    def predict(
        self,
        X_test: torch.Tensor,
        use_log_transform: bool = True,
        clip_values: bool = False,
        min_calories: Optional[float] = None,
        max_calories: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate predictions using ensemble of models.
        
        Args:
            X_test: Test features tensor
            use_log_transform: Whether to inverse log transform predictions
            clip_values: Whether to clip predictions to realistic range
            min_calories: Minimum calorie value for clipping (ignored if clip_values=False)
            max_calories: Maximum calorie value for clipping (ignored if clip_values=False)
        
        Returns:
            Array of predictions
        """
        predictions = []
        
        with torch.no_grad():
            X_test_gpu = X_test.to(self.device)
            
            for model in self.models:
                pred = model(X_test_gpu).cpu().numpy().flatten()
                predictions.append(pred)
        
        ensemble_pred = np.mean(predictions, axis=0)
        
        if use_log_transform:
            ensemble_pred = np.expm1(ensemble_pred)
        
        if clip_values and min_calories is not None and max_calories is not None:
            ensemble_pred = np.clip(ensemble_pred, min_calories, max_calories)
        
        return ensemble_pred
    
    def predict_from_file(
        self,
        test_path: str,
        train_path: Optional[str] = None,
        use_log_transform: bool = True,
        clip_values: bool = False
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Generate predictions directly from test file.
        
        Args:
            test_path: Path to test CSV
            train_path: Optional path to training CSV for normalization
            use_log_transform: Whether to inverse log transform predictions
            clip_values: Whether to clip predictions to realistic range
        
        Returns:
            Tuple of (predictions array, test dataframe)
        """
        X_test, test_df = self.prepare_test_data(test_path, train_path)
        
        predictions = self.predict(
            X_test,
            use_log_transform=use_log_transform,
            clip_values=clip_values
        )
        
        return predictions, test_df
    
    def create_submission(
        self,
        test_path: str,
        output_path: str,
        train_path: Optional[str] = None,
        use_log_transform: bool = True,
        clip_values: bool = False
    ) -> None:
        """
        Create competition submission file.
        
        Args:
            test_path: Path to test CSV
            output_path: Path to save submission CSV
            train_path: Optional path to training CSV for normalization
            use_log_transform: Whether to inverse log transform predictions
            clip_values: Whether to clip predictions to realistic range
        """
        predictions, test_df = self.predict_from_file(
            test_path,
            train_path,
            use_log_transform,
            clip_values
        )
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Calories': predictions
        })
        
        submission.to_csv(output_path, index=False)
        
        print(f"\nSubmission file created: {output_path}")
        print(f"Number of predictions: {len(submission)}")
        print(f"Prediction statistics:")
        print(f"  Mean: {predictions.mean():.2f}")
        print(f"  Median: {np.median(predictions):.2f}")
        print(f"  Std: {predictions.std():.2f}")
        print(f"  Min: {predictions.min():.2f}")
        print(f"  Max: {predictions.max():.2f}")


def create_submission_from_experiment(
    experiment_dir: str,
    test_path: str,
    train_path: str,
    output_path: str,
    architecture: str = 'simple',
    dropout: float = 0.0,
    **model_kwargs
) -> None:
    """
    Create submission using all fold models from an experiment.
    
    Args:
        experiment_dir: Directory containing fold model checkpoints
        test_path: Path to test CSV
        train_path: Path to training CSV
        output_path: Path to save submission CSV
        architecture: Model architecture name
        dropout: Dropout rate used in training
        **model_kwargs: Additional model parameters
    """
    exp_path = Path(experiment_dir)
    model_paths = sorted(exp_path.glob("model_fold_*.pth"))
    
    if not model_paths:
        raise FileNotFoundError(f"No model checkpoints found in {experiment_dir}")
    
    model_paths_str = [str(p) for p in model_paths]
    
    print(f"Creating submission using {len(model_paths_str)} models from {experiment_dir}")
    
    predictor = CaloriePredictor(
        model_paths=model_paths_str,
        architecture=architecture,
        dropout=dropout,
        **model_kwargs
    )
    
    predictor.create_submission(
        test_path=test_path,
        output_path=output_path,
        train_path=train_path,
        use_log_transform=True,
        clip_values=False
    )


def create_best_submission(
    experiments_dir: str,
    test_path: str,
    train_path: str,
    output_path: str
) -> None:
    """
    Automatically find best experiment and create submission.
    
    Args:
        experiments_dir: Root directory containing all experiments
        test_path: Path to test CSV
        train_path: Path to training CSV
        output_path: Path to save submission CSV
    """
    import json
    
    exp_root = Path(experiments_dir)
    comparison_file = exp_root / "comparison_summary.json"
    
    if not comparison_file.exists():
        raise FileNotFoundError(
            f"Comparison summary not found. Run model comparison first."
        )
    
    with open(comparison_file, 'r') as f:
        summary = json.load(f)
    
    best_exp_name = summary['best_experiment']
    best_rmse = summary['best_rmse']
    
    print(f"Best experiment: {best_exp_name}")
    print(f"Best RMSE: {best_rmse:.5f}")
    
    best_exp = next(
        exp for exp in summary['experiments'] 
        if exp['name'] == best_exp_name
    )
    
    architecture = best_exp['config']['architecture']
    dropout = best_exp['config']['dropout']
    
    experiment_dir = exp_root / best_exp_name
    
    create_submission_from_experiment(
        experiment_dir=str(experiment_dir),
        test_path=test_path,
        train_path=train_path,
        output_path=output_path,
        architecture=architecture,
        dropout=dropout
    )
