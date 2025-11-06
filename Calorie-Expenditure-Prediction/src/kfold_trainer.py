"""
Advanced training utilities with KFold cross-validation, early stopping, and learning rate scheduling.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from rmsle_loss import RMSLELoss


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    patience: int = 10
    lr_factor: float = 0.5
    lr_patience: int = 3
    min_lr: float = 1e-6
    n_folds: int = 5
    random_state: int = 42
    use_log_transform: bool = True
    normalize_features: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """
        Create TrainingConfig from dictionary (e.g., from YAML config).
        
        Args:
            config_dict: Dictionary with training configuration
        
        Returns:
            TrainingConfig instance
        """
        return cls(
            epochs=config_dict.get('epochs', 100),
            batch_size=config_dict.get('batch_size', 256),
            learning_rate=config_dict.get('learning_rate', 0.001),
            weight_decay=config_dict.get('weight_decay', 0.0),
            patience=config_dict.get('patience', 10),
            lr_factor=config_dict.get('lr_factor', 0.5),
            lr_patience=config_dict.get('lr_patience', 3),
            min_lr=config_dict.get('min_lr', 1e-6),
            n_folds=config_dict.get('n_folds', 5),
            random_state=config_dict.get('random_state', 42),
            use_log_transform=config_dict.get('use_log_transform', True),
            normalize_features=config_dict.get('normalize_features', True)
        )


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    """
    
    def __init__(
        self, 
        patience: int = 20, 
        min_delta: float = 0.0,
        mode: str = 'min'
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_model_state: Optional[Dict] = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop


class LRSchedulerWrapper:
    """
    Learning rate scheduler with plateau reduction.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        factor: float = 0.5,
        patience: int = 3,
        min_lr: float = 1e-6,
        verbose: bool = True
    ) -> None:
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.counter = 0
        self.best_loss: Optional[float] = None
    
    def step(self, val_loss: float) -> None:
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
    
    def _reduce_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}")


class KFoldTrainer:
    """
    KFold cross-validation trainer with comprehensive metrics tracking.
    """
    
    def __init__(
        self,
        model_factory: Callable,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ) -> None:
        self.model_factory = model_factory
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.fold_results: List[Dict[str, Any]] = []
        self.oof_predictions: Optional[np.ndarray] = None
        self.test_predictions: Optional[np.ndarray] = None
    
    def prepare_data(
        self,
        train_path: str,
        test_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Load and prepare data for training.
        
        Args:
            train_path: Path to training CSV
            test_path: Optional path to test CSV
        
        Returns:
            Tuple of (train_features, train_target, test_features)
        """
        train_df = pd.read_csv(train_path)
        train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1}).astype('float32')
        
        features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        
        X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
        y_train = torch.tensor(train_df['Calories'].values, dtype=torch.float32).view(-1, 1)
        
        if self.config.use_log_transform:
            y_train = torch.log1p(y_train)
        
        X_test = None
        if test_path:
            test_df = pd.read_csv(test_path)
            test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1}).astype('float32')
            X_test = torch.tensor(test_df[features].values, dtype=torch.float32)
        
        return X_train, y_train, X_test
    
    def normalize_features(
        self,
        X_train: torch.Tensor,
        X_valid: torch.Tensor,
        X_test: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Normalize features using training set statistics.
        
        Args:
            X_train: Training features
            X_valid: Validation features
            X_test: Optional test features
        
        Returns:
            Tuple of normalized tensors
        """
        means = X_train.mean(dim=0, keepdim=True)
        stds = X_train.std(dim=0, keepdim=True)
        stds = torch.where(stds == 0, torch.ones_like(stds), stds)
        
        X_train_norm = (X_train - means) / stds
        X_valid_norm = (X_valid - means) / stds
        
        X_test_norm = None
        if X_test is not None:
            X_test_norm = (X_test - means) / stds
        
        return X_train_norm, X_valid_norm, X_test_norm
    
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> float:
        """
        Train model for one epoch.
        
        Args:
            model: Neural network model
            dataloader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
        
        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(
        self,
        model: nn.Module,
        X_valid: torch.Tensor,
        y_valid: torch.Tensor,
        criterion: nn.Module
    ) -> Tuple[float, np.ndarray]:
        """
        Validate model on validation set.
        
        Args:
            model: Neural network model
            X_valid: Validation features
            y_valid: Validation targets
            criterion: Loss function
        
        Returns:
            Tuple of (validation loss, predictions)
        """
        model.eval()
        with torch.no_grad():
            X_valid = X_valid.to(self.device)
            y_valid_gpu = y_valid.to(self.device)
            
            predictions = model(X_valid)
            predictions = torch.clamp(predictions, min=0) 
            loss = criterion(predictions, y_valid_gpu)
            
            predictions_np = predictions.cpu().numpy().flatten()
        
        return loss.item(), predictions_np
    
    def train_fold(
        self,
        fold: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_valid: torch.Tensor,
        y_valid: torch.Tensor,
        X_test: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Train model on a single fold.
        
        Args:
            fold: Fold number
            X_train: Training features
            y_train: Training targets
            X_valid: Validation features
            y_valid: Validation targets
            X_test: Optional test features
        
        Returns:
            Dictionary with fold results
        """
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                        FOLD {fold + 1:2d} START                           ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        if self.config.normalize_features:
            X_train, X_valid, X_test = self.normalize_features(X_train, X_valid, X_test)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        model = self.model_factory().to(self.device)
        criterion = RMSLELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        early_stopping = EarlyStopping(patience=self.config.patience)
        lr_scheduler = LRSchedulerWrapper(
            optimizer,
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            min_lr=self.config.min_lr
        )
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_preds = self.validate(model, X_valid, y_valid, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            lr_scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.config.epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if early_stopping(val_loss, model):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                model.load_state_dict(early_stopping.best_model_state)
                break
        
        _, oof_preds = self.validate(model, X_valid, y_valid, criterion)
        
        test_preds = None
        if X_test is not None:
            model.eval()
            with torch.no_grad():
                X_test = X_test.to(self.device)
                test_preds = model(X_test)
                test_preds = torch.clamp(test_preds, min=0).cpu().numpy().flatten()
        
        y_valid_np = y_valid.numpy().flatten()
        rmse = np.sqrt(mean_squared_error(y_valid_np, oof_preds))
        mae = mean_absolute_error(y_valid_np, oof_preds)
        r2 = r2_score(y_valid_np, oof_preds)
        
        y_valid_actual = np.expm1(y_valid_np) if self.config.use_log_transform else y_valid_np
        oof_preds_actual = np.expm1(oof_preds) if self.config.use_log_transform else oof_preds
        mape = mean_absolute_percentage_error(y_valid_actual, oof_preds_actual)
        
        rmsle = np.sqrt(mean_squared_error(np.log(oof_preds_actual + 1), np.log(y_valid_actual + 1)))
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                     FOLD {fold + 1} RESULTS                           ║
╠══════════════════════════════════════════════════════════════╣
║  RMSE:  {rmse:>8.4f}    MAE:  {mae:>8.4f}    R2:  {r2:>8.4f}  ║
║  RMSLE: {rmsle:>8.4f}  MAPE: {mape:>8.4f} ({mape*100:>6.2f}%)         ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        return {
            'fold': fold + 1,
            'rmse': rmse,
            'mae': mae,
            'rmsle': rmsle,
            'mape': mape,
            'r2': r2,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'oof_predictions': oof_preds,
            'test_predictions': test_preds,
            'model_state': model.state_dict()
        }
    
    def train_kfold(
        self,
        train_path: str,
        test_path: Optional[str] = None,
        save_dir: str = "models"
    ) -> Dict[str, Any]:
        """
        Perform KFold cross-validation training.
        
        Args:
            train_path: Path to training data
            test_path: Optional path to test data
            save_dir: Directory to save models and results
        
        Returns:
            Dictionary with overall results
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        X_train, y_train, X_test = self.prepare_data(train_path, test_path)
        
        kf = KFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        self.oof_predictions = np.zeros(len(X_train))
        if X_test is not None:
            self.test_predictions = np.zeros(len(X_test))
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_valid_fold = X_train[valid_idx]
            y_valid_fold = y_train[valid_idx]
            
            fold_result = self.train_fold(
                fold,
                X_train_fold,
                y_train_fold,
                X_valid_fold,
                y_valid_fold,
                X_test
            )
            
            self.oof_predictions[valid_idx] = fold_result['oof_predictions']
            if X_test is not None and fold_result['test_predictions'] is not None:
                self.test_predictions += fold_result['test_predictions']
            
            torch.save(
                fold_result['model_state'],
                save_path / f"model_fold_{fold + 1}.pth"
            )
            
            self.fold_results.append(fold_result)
        
        if X_test is not None:
            self.test_predictions /= self.config.n_folds
        
        overall_rmse = np.sqrt(mean_squared_error(
            y_train.numpy().flatten(),
            self.oof_predictions
        ))
        
        y_train_actual = np.expm1(y_train.numpy().flatten()) if self.config.use_log_transform else y_train.numpy().flatten()
        oof_preds_actual = np.expm1(self.oof_predictions) if self.config.use_log_transform else self.oof_predictions
        overall_mape = mean_absolute_percentage_error(y_train_actual, oof_preds_actual)
        overall_rmsle = np.sqrt(mean_squared_error(np.log(oof_preds_actual + 1), np.log(y_train_actual + 1)))
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              OVERALL CROSS-VALIDATION RESULTS               ║
╠══════════════════════════════════════════════════════════════╣
║  Overall RMSE:  {overall_rmse:>8.5f}                           ║
║  Overall RMSLE: {overall_rmsle:>8.5f}                           ║
║  Overall MAPE:  {overall_mape:>8.5f} ({overall_mape*100:>6.2f}%)           ║
╠══════════════════════════════════════════════════════════════╣
║                    FOLD-WISE RESULTS                        ║
╚══════════════════════════════════════════════════════════════╝
""")
        for result in self.fold_results:
            print(f"  Fold {result['fold']}: RMSE={result['rmse']:.5f}, MAPE={result['mape']:.5f}")
        
        results = {
            'overall_rmse': overall_rmse,
            'overall_rmsle': overall_rmsle,
            'overall_mape': overall_mape,
            'fold_results': self.fold_results,
            'oof_predictions': self.oof_predictions,
            'test_predictions': self.test_predictions,
            'config': self.config.__dict__
        }
        
        with open(save_path / "training_results.json", 'w') as f:
            json_results = {
                'overall_rmse': overall_rmse,
                'overall_rmsle': overall_rmsle,
                'overall_mape': overall_mape,
                'fold_results': [
                    {k: v for k, v in r.items() 
                     if k not in ['oof_predictions', 'test_predictions', 'model_state', 'train_losses', 'val_losses']}
                    for r in self.fold_results
                ],
                'config': self.config.__dict__
            }
            json.dump(json_results, f, indent=2)
        
        return results
