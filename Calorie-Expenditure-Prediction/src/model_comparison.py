"""
Comprehensive model comparison framework for architecture and hyperparameter evaluation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from architectures import get_model, count_parameters
from kfold_trainer import KFoldTrainer, TrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    architecture: str
    dropout: float = 0.0
    learning_rate: float = 0.001
    batch_size: int = 256
    weight_decay: float = 0.0
    hidden_size: Optional[int] = None
    num_blocks: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ModelComparator:
    """
    Framework for comparing multiple model architectures and hyperparameters.
    """
    
    def __init__(
        self,
        train_path: str,
        test_path: Optional[str] = None,
        base_config: Optional[TrainingConfig] = None,
        output_dir: str = "experiments"
    ) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.base_config = base_config or TrainingConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.experiments: List[Dict[str, Any]] = []
    
    def create_model_factory(self, exp_config: ExperimentConfig):
        """
        Create model factory function for a specific experiment configuration.
        
        Args:
            exp_config: Experiment configuration
        
        Returns:
            Factory function that creates model instances
        """
        def factory():
            kwargs = {}
            if exp_config.hidden_size is not None:
                kwargs['hidden_size'] = exp_config.hidden_size
            if exp_config.num_blocks is not None:
                kwargs['num_blocks'] = exp_config.num_blocks
            
            model = get_model(
                exp_config.architecture,
                input_size=7,
                dropout=exp_config.dropout,
                **kwargs
            )
            return model
        return factory
    
    def run_experiment(
        self,
        exp_config: ExperimentConfig,
        training_config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """
        Run a single experiment with specified configuration.
        
        Args:
            exp_config: Experiment configuration
            training_config: Optional training configuration (uses base_config if None)
        
        Returns:
            Dictionary with experiment results
        """
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    STARTING EXPERIMENT                      ║
║                                                              ║
║  Name: {exp_config.name:<20} Architecture: {exp_config.architecture:<10}  ║
║  Dropout: {exp_config.dropout:<5} Learning Rate: {exp_config.learning_rate:<8}  ║
║  Batch Size: {exp_config.batch_size:<3}                                     ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        config = training_config or TrainingConfig()
        config.learning_rate = exp_config.learning_rate
        config.batch_size = exp_config.batch_size
        config.weight_decay = exp_config.weight_decay
        
        model_factory = self.create_model_factory(exp_config)
        sample_model = model_factory()
        num_params = count_parameters(sample_model)
        print(f"Model Parameters: {num_params:,}")
        
        exp_output_dir = self.output_dir / exp_config.name
        exp_output_dir.mkdir(exist_ok=True)
        
        trainer = KFoldTrainer(
            model_factory=model_factory,
            config=config
        )
        
        start_time = time.time()
        results = trainer.train_kfold(
            train_path=self.train_path,
            test_path=self.test_path,
            save_dir=str(exp_output_dir)
        )
        training_time = time.time() - start_time
        
        experiment_result = {
            'name': exp_config.name,
            'config': exp_config.to_dict(),
            'num_parameters': num_params,
            'overall_rmse': results['overall_rmse'],
            'overall_rmsle': results['overall_rmsle'],
            'overall_mape': results['overall_mape'],
            'fold_results': [
                {k: v for k, v in r.items() 
                 if k not in ['oof_predictions', 'test_predictions', 'model_state', 'train_losses', 'val_losses']}
                for r in results['fold_results']
            ],
            'training_time': training_time,
            'oof_predictions': results['oof_predictions'],
            'test_predictions': results['test_predictions']
        }
        
        self.experiments.append(experiment_result)
        
        with open(exp_output_dir / "experiment_results.json", 'w') as f:
            json_result = {k: v for k, v in experiment_result.items() 
                          if k not in ['oof_predictions', 'test_predictions']}
            json.dump(json_result, f, indent=2)
        
        np.save(exp_output_dir / "oof_predictions.npy", results['oof_predictions'])
        if results['test_predictions'] is not None:
            np.save(exp_output_dir / "test_predictions.npy", results['test_predictions'])
        
        print(f"\nExperiment completed in {training_time:.2f} seconds")
        
        return experiment_result
    
    def compare_architectures(
        self,
        architectures: List[str],
        dropout_rates: List[float] = [0.0],
        learning_rates: List[float] = [0.001]
    ) -> None:
        """
        Compare different architectures with various hyperparameters.
        
        Args:
            architectures: List of architecture names to compare
            dropout_rates: List of dropout rates to test
            learning_rates: List of learning rates to test
        """
        for arch in architectures:
            for dropout in dropout_rates:
                for lr in learning_rates:
                    dropout_str = f"_dropout{dropout}" if dropout > 0 else "_nodropout"
                    exp_name = f"{arch}{dropout_str}_lr{lr}"
                    
                    exp_config = ExperimentConfig(
                        name=exp_name,
                        architecture=arch,
                        dropout=dropout,
                        learning_rate=lr,
                        batch_size=self.base_config.batch_size
                    )
                    
                    self.run_experiment(exp_config, self.base_config)
    
    def compare_hyperparameters(
        self,
        architecture: str = 'simple',
        batch_sizes: List[int] = [128, 256, 512],
        learning_rates: List[float] = [0.0001, 0.001, 0.01],
        dropout_rate: float = 0.0
    ) -> None:
        """
        Compare different hyperparameter configurations for a single architecture.
        
        Args:
            architecture: Architecture to use
            batch_sizes: List of batch sizes to test
            learning_rates: List of learning rates to test
            dropout_rate: Dropout rate to use
        """
        for batch_size in batch_sizes:
            for lr in learning_rates:
                exp_name = f"{architecture}_bs{batch_size}_lr{lr}"
                
                exp_config = ExperimentConfig(
                    name=exp_name,
                    architecture=architecture,
                    dropout=dropout_rate,
                    learning_rate=lr,
                    batch_size=batch_size
                )
                
                self.run_experiment(exp_config)
    
    def generate_comparison_report(self, save_path: Optional[str] = None) -> None:
        """
        Generate comprehensive comparison report with visualizations.
        
        Args:
            save_path: Optional path to save the report plots
        """
        if not self.experiments:
            print("No experiments to compare!")
            return
        
        print(f"\n{'='*70}")
        print("MODEL COMPARISON REPORT")
        print(f"{'='*70}\n")
        
        sorted_experiments = sorted(
            self.experiments,
            key=lambda x: x['overall_mape']
        )
        
        print("Results Summary (sorted by MAPE):")
        print("-" * 70)
        print(f"{'Rank':<6} {'Experiment':<30} {'MAPE':<12} {'RMSE':<12} {'Params':<12} {'Time (s)':<10}")
        print("-" * 70)
        
        for rank, exp in enumerate(sorted_experiments, 1):
            print(f"{rank:<6} {exp['name']:<30} {exp['overall_mape']:<12.5f} "
                  f"{exp['overall_rmse']:<12.5f} {exp['num_parameters']:<12,} {exp['training_time']:<10.1f}")
        
        print("-" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        names = [exp['name'] for exp in sorted_experiments[:10]]
        mapes = [exp['overall_mape'] for exp in sorted_experiments[:10]]
        
        axes[0, 0].barh(names, mapes)
        axes[0, 0].set_xlabel('MAPE')
        axes[0, 0].set_title('Top 10 Models by MAPE (Lower is Better)')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, alpha=0.3)
        
        params = [exp['num_parameters'] for exp in self.experiments]
        mapes_all = [exp['overall_mape'] for exp in self.experiments]
        
        axes[0, 1].scatter(params, mapes_all, alpha=0.6, s=100)
        axes[0, 1].set_xlabel('Number of Parameters')
        axes[0, 1].set_ylabel('MAPE')
        axes[0, 1].set_title('Model Complexity vs Performance')
        axes[0, 1].grid(True, alpha=0.3)
        
        for i, exp in enumerate(self.experiments):
            axes[0, 1].annotate(
                exp['name'], 
                (params[i], mapes_all[i]),
                fontsize=7,
                alpha=0.7
            )
        
        train_times = [exp['training_time'] for exp in self.experiments]
        
        axes[1, 0].scatter(train_times, mapes_all, alpha=0.6, s=100, color='green')
        axes[1, 0].set_xlabel('Training Time (seconds)')
        axes[1, 0].set_ylabel('MAPE')
        axes[1, 0].set_title('Training Time vs Performance')
        axes[1, 0].grid(True, alpha=0.3)
        
        dropout_exp = [exp for exp in self.experiments if exp['config']['dropout'] > 0]
        nodropout_exp = [exp for exp in self.experiments if exp['config']['dropout'] == 0]
        
        if dropout_exp and nodropout_exp:
            dropout_mapes = [exp['overall_mape'] for exp in dropout_exp]
            nodropout_mapes = [exp['overall_mape'] for exp in nodropout_exp]
            
            axes[1, 1].boxplot(
                [nodropout_mapes, dropout_mapes],
                labels=['No Dropout', 'With Dropout']
            )
            axes[1, 1].set_ylabel('MAPE')
            axes[1, 1].set_title('Dropout Impact on Performance (Lower is Better)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5, 0.5,
                'Insufficient data for dropout comparison',
                ha='center',
                va='center',
                transform=axes[1, 1].transAxes
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        summary_path = self.output_dir / "comparison_summary.json"
        with open(summary_path, 'w') as f:
            summary = {
                'num_experiments': len(self.experiments),
                'best_experiment': sorted_experiments[0]['name'],
                'best_mape': sorted_experiments[0]['overall_mape'],
                'best_rmse': sorted_experiments[0]['overall_rmse'],
                'experiments': [
                    {k: v for k, v in exp.items() 
                     if k not in ['oof_predictions', 'test_predictions']}
                    for exp in sorted_experiments
                ]
            }
            json.dump(summary, f, indent=2)
        
        print(f"\nComparison summary saved to: {summary_path}")


def run_default_comparison(
    train_path: str,
    test_path: Optional[str] = None,
    output_dir: str = "experiments",
    config_path: str = "config/config.yaml"
) -> ModelComparator:
    """
    Run default comparison suite across architectures and hyperparameters.
    
    Args:
        train_path: Path to training data
        test_path: Optional path to test data
        output_dir: Directory for experiment outputs
        config_path: Path to configuration YAML file
    
    Returns:
        ModelComparator instance with results
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_config = TrainingConfig.from_dict(config['training'])
    
    comparator = ModelComparator(
        train_path=train_path,
        test_path=test_path,
        base_config=base_config,
        output_dir=output_dir
    )
    
    print("Starting comprehensive model comparison...")
    
    architectures = config.get('experiments', {}).get('architectures', ['simple', 'deep', 'wide', 'residual', 'adaptive'])
    dropout_rates = config.get('experiments', {}).get('dropout_rates', [0.0, 0.2])
    learning_rates = config.get('experiments', {}).get('learning_rates', [0.001])
    
    comparator.compare_architectures(
        architectures=architectures,
        dropout_rates=dropout_rates,
        learning_rates=learning_rates
    )
    
    print("\nRunning hyperparameter comparison for best architecture...")
    batch_sizes = config.get('experiments', {}).get('batch_sizes', [256])
    comparator.compare_hyperparameters(
        architecture='simple',
        batch_sizes=batch_sizes,
        learning_rates=learning_rates,
        dropout_rate=0.0
    )
    
    comparator.generate_comparison_report(
        save_path=str(Path(output_dir) / "comparison_report.png")
    )
    
    return comparator
