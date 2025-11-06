"""
Main script for running model training and comparison experiments.
"""

import sys
from pathlib import Path
import argparse
import yaml

sys.path.append(str(Path(__file__).parent))

from model_comparison import run_default_comparison, ModelComparator, ExperimentConfig
from kfold_trainer import TrainingConfig


def main() -> None:
    """
    Run model training and comparison pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Train and compare neural network models for calorie prediction"
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default='config/config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['default', 'quick', 'custom'],
        default='default',
        help='Experiment mode: default (comprehensive), quick (fast test), custom (manual config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=None,
        help='Number of cross-validation folds (overrides config)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CALORIE EXPENDITURE PREDICTION - MODEL TRAINING & COMPARISON")
    print("=" * 70)
    
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    training_config = config['training']
    experiments_config = config.get('experiments', {})
    
    if args.epochs is not None:
        training_config['epochs'] = args.epochs
    if args.folds is not None:
        training_config['n_folds'] = args.folds
    
    base_config = TrainingConfig.from_dict(training_config)
    
    print(f"\nConfiguration loaded from: {args.config_path}")
    print(f"  Train path: {data_config['train']}")
    print(f"  Test path: {data_config['test']}")
    print(f"  Output directory: {experiments_config.get('output_dir', 'experiments')}")
    print(f"  Mode: {args.mode}")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Folds: {training_config['n_folds']}")
    print()
    
    if args.mode == 'default':
        print("Running comprehensive model comparison...")
        comparator = run_default_comparison(
            train_path=data_config['train'],
            test_path=data_config['test'],
            output_dir=experiments_config.get('output_dir', 'experiments'),
            config_path=args.config_path
        )
    
    elif args.mode == 'quick':
        print("Running quick comparison with limited configurations...")
        comparator = ModelComparator(
            train_path=data_config['train'],
            test_path=data_config['test'],
            base_config=base_config,
            output_dir=experiments_config.get('output_dir', 'experiments')
        )
        
        architectures = experiments_config.get('architectures', ['simple', 'deep'])[:2]  # Only first 2
        dropout_rates = experiments_config.get('dropout_rates', [0.0, 0.2])
        learning_rates = experiments_config.get('learning_rates', [0.001])
        
        comparator.compare_architectures(
            architectures=architectures,
            dropout_rates=dropout_rates,
            learning_rates=learning_rates
        )
        
        comparator.generate_comparison_report(
            save_path=str(Path(experiments_config.get('output_dir', 'experiments')) / "comparison_report.png")
        )
    
    elif args.mode == 'custom':
        print("Running custom configuration...")
        comparator = ModelComparator(
            train_path=data_config['train'],
            test_path=data_config['test'],
            base_config=base_config,
            output_dir=experiments_config.get('output_dir', 'experiments')
        )
        
        exp_config = ExperimentConfig(
            name='custom_experiment',
            architecture='simple',
            dropout=0.2,
            learning_rate=0.001,
            batch_size=training_config['batch_size']
        )
        
        comparator.run_experiment(exp_config)
        comparator.generate_comparison_report()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
