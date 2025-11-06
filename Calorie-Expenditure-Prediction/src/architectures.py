"""
Neural network architectures for calorie expenditure prediction.
Contains multiple architecture variants for comparison and experimentation.
"""

from typing import Optional
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """
    Simple baseline architecture with 3 hidden layers.
    This is the original NeuralNetwork architecture renamed for clarity.
    """
    
    def __init__(self, input_size: int = 7, dropout: float = 0.0) -> None:
        super(SimpleNet, self).__init__()
        self.use_dropout = dropout > 0.0
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepNet(nn.Module):
    """
    Deeper architecture with 5 hidden layers for increased capacity.
    """
    
    def __init__(self, input_size: int = 7, dropout: float = 0.0) -> None:
        super(DeepNet, self).__init__()
        self.use_dropout = dropout > 0.0
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WideNet(nn.Module):
    """
    Wide architecture with larger layer sizes but fewer layers.
    """
    
    def __init__(self, input_size: int = 7, dropout: float = 0.0) -> None:
        super(WideNet, self).__init__()
        self.use_dropout = dropout > 0.0
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for deeper networks.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super(ResidualBlock, self).__init__()
        self.use_dropout = dropout > 0.0
        
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out += residual
        out = self.activation(out)
        return out


class ResidualNet(nn.Module):
    """
    Residual network with skip connections for better gradient flow.
    """
    
    def __init__(
        self, 
        input_size: int = 7, 
        hidden_size: int = 64, 
        num_blocks: int = 3,
        dropout: float = 0.0
    ) -> None:
        super(ResidualNet, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) 
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.output_layer(x)
        return x


class AdaptiveNet(nn.Module):
    """
    Adaptive architecture with attention-like mechanism.
    """
    
    def __init__(self, input_size: int = 7, dropout: float = 0.0) -> None:
        super(AdaptiveNet, self).__init__()
        self.use_dropout = dropout > 0.0
        
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
        
        self.main_path = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout) if self.use_dropout else nn.Identity(),
        )
        
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.feature_attention(x)
        x_weighted = x * attention_weights
        
        x = self.main_path(x_weighted)
        x = self.output_layer(x)
        return x


def get_model(
    architecture: str, 
    input_size: int = 7, 
    dropout: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create model instances by architecture name.
    
    Args:
        architecture: Name of the architecture ('simple', 'deep', 'wide', 'residual', 'adaptive')
        input_size: Number of input features
        dropout: Dropout rate (0.0 means no dropout)
        **kwargs: Additional architecture-specific parameters
    
    Returns:
        Initialized model instance
    """
    architectures = {
        'simple': SimpleNet,
        'deep': DeepNet,
        'wide': WideNet,
        'residual': ResidualNet,
        'adaptive': AdaptiveNet,
    }
    
    if architecture.lower() not in architectures:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: {list(architectures.keys())}"
        )
    
    model_class = architectures[architecture.lower()]
    return model_class(input_size=input_size, dropout=dropout, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
