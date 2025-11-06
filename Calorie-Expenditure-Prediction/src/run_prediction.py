"""
Main script for generating predictions and submission files.
"""

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent))

from predict import create_best_submission, create_submission_from_experiment


def main() -> None:
    """
    Generate predictions and create submission file.
    """
    parser = argparse.ArgumentParser(
        description="Generate predictions for calorie expenditure"
    )
    parser.add_argument(
        '--train-path',
        type=str,
        default='data/train.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--test-path',
        type=str,
        default='data/test.csv',
        help='Path to test data CSV'
    )
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='experiments',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='submission.csv',
        help='Path to save submission CSV'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Specific experiment name to use (default: best experiment)'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='simple',
        help='Architecture name (required if using --experiment-name)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout rate (required if using --experiment-name)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CALORIE EXPENDITURE PREDICTION - SUBMISSION GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Train path: {args.train_path}")
    print(f"  Test path: {args.test_path}")
    print(f"  Experiments directory: {args.experiments_dir}")
    print(f"  Output path: {args.output_path}")
    print()
    
    if args.experiment_name:
        print(f"Using specific experiment: {args.experiment_name}")
        experiment_dir = Path(args.experiments_dir) / args.experiment_name
        
        if not experiment_dir.exists():
            print(f"Error: Experiment directory not found: {experiment_dir}")
            return
        
        create_submission_from_experiment(
            experiment_dir=str(experiment_dir),
            test_path=args.test_path,
            train_path=args.train_path,
            output_path=args.output_path,
            architecture=args.architecture,
            dropout=args.dropout
        )
    else:
        print("Automatically selecting best experiment...")
        create_best_submission(
            experiments_dir=args.experiments_dir,
            test_path=args.test_path,
            train_path=args.train_path,
            output_path=args.output_path
        )
    
    print("\n" + "=" * 70)
    print("SUBMISSION GENERATED SUCCESSFULLY!")
    print(f"File saved to: {args.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
