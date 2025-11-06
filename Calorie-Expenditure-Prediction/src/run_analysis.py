"""
Main script for running comprehensive data analysis.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_analysis import CalorieDataAnalyzer


def main() -> None:
    """
    Run complete data analysis pipeline.
    """
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    output_dir = "analysis_results"
    
    print("Starting Calorie Expenditure Data Analysis...")
    print("=" * 70)
    
    analyzer = CalorieDataAnalyzer(
        train_path=train_path,
        test_path=test_path
    )
    
    analyzer.generate_full_report(output_dir=output_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
