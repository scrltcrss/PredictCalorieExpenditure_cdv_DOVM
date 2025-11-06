"""
Exploratory Data Analysis and visualization utilities for calorie prediction dataset.
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class CalorieDataAnalyzer:
    """
    Comprehensive data analysis toolkit for workout calorie prediction.
    """
    
    def __init__(self, train_path: str, test_path: Optional[str] = None) -> None:
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path) if test_path else None
        self._prepare_data()
    
    def _prepare_data(self) -> None:
        """Encode categorical features for analysis."""
        for df in [self.train_df, self.test_df]:
            if df is not None:
                df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
    
    def get_basic_statistics(self) -> pd.DataFrame:
        """
        Generate comprehensive statistics for all features.
        
        Returns:
            DataFrame with descriptive statistics
        """
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
        stats = self.train_df[numeric_cols].describe()
        
        stats.loc['missing'] = self.train_df[numeric_cols].isnull().sum()
        stats.loc['skewness'] = self.train_df[numeric_cols].skew()
        stats.loc['kurtosis'] = self.train_df[numeric_cols].kurtosis()
        
        return stats
    
    def analyze_target_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Analyze and visualize target variable distribution.
        
        Args:
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(self.train_df['Calories'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title('Calories Distribution')
        axes[0].set_xlabel('Calories')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(np.log1p(self.train_df['Calories']), bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_title('Log-transformed Calories Distribution')
        axes[1].set_xlabel('log(Calories + 1)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        from scipy import stats
        stats.probplot(self.train_df['Calories'], dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Target Statistics:")
        print(f"Mean: {self.train_df['Calories'].mean():.2f}")
        print(f"Median: {self.train_df['Calories'].median():.2f}")
        print(f"Std: {self.train_df['Calories'].std():.2f}")
        print(f"Min: {self.train_df['Calories'].min():.2f}")
        print(f"Max: {self.train_df['Calories'].max():.2f}")
        print(f"Skewness: {self.train_df['Calories'].skew():.2f}")
    
    def plot_feature_distributions(self, save_path: Optional[str] = None) -> None:
        """
        Visualize distributions of all numeric features.
        
        Args:
            save_path: Optional path to save the figure
        """
        numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, feature in enumerate(numeric_features):
            axes[idx].hist(self.train_df[feature], bins=40, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{feature} Distribution')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, save_path: Optional[str] = None) -> None:
        """
        Generate correlation heatmap for numeric features.
        
        Args:
            save_path: Optional path to save the figure
        """
        numeric_cols = ['Sex_encoded', 'Age', 'Height', 'Weight', 
                       'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
        
        corr_matrix = self.train_df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nTop correlations with Calories:")
        calories_corr = corr_matrix['Calories'].drop('Calories').sort_values(ascending=False)
        for feature, corr in calories_corr.items():
            print(f"{feature}: {corr:.3f}")
    
    def analyze_gender_differences(self, save_path: Optional[str] = None) -> None:
        """
        Compare calorie expenditure patterns between genders.
        
        Args:
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        male_data = self.train_df[self.train_df['Sex'] == 'male']
        female_data = self.train_df[self.train_df['Sex'] == 'female']
        
        axes[0, 0].hist(male_data['Calories'], bins=40, alpha=0.6, label='Male', color='blue')
        axes[0, 0].hist(female_data['Calories'], bins=40, alpha=0.6, label='Female', color='red')
        axes[0, 0].set_title('Calories Distribution by Gender')
        axes[0, 0].set_xlabel('Calories')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        gender_stats = self.train_df.groupby('Sex')['Calories'].agg(['mean', 'median', 'std'])
        gender_stats.plot(kind='bar', ax=axes[0, 1], rot=0)
        axes[0, 1].set_title('Calorie Statistics by Gender')
        axes[0, 1].set_ylabel('Calories')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].scatter(male_data['Weight'], male_data['Calories'], 
                          alpha=0.3, label='Male', color='blue', s=10)
        axes[1, 0].scatter(female_data['Weight'], female_data['Calories'], 
                          alpha=0.3, label='Female', color='red', s=10)
        axes[1, 0].set_title('Calories vs Weight by Gender')
        axes[1, 0].set_xlabel('Weight')
        axes[1, 0].set_ylabel('Calories')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(male_data['Duration'], male_data['Calories'], 
                          alpha=0.3, label='Male', color='blue', s=10)
        axes[1, 1].scatter(female_data['Duration'], female_data['Calories'], 
                          alpha=0.3, label='Female', color='red', s=10)
        axes[1, 1].set_title('Calories vs Duration by Gender')
        axes[1, 1].set_xlabel('Duration')
        axes[1, 1].set_ylabel('Calories')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nGender-based statistics:")
        print(gender_stats)
    
    def analyze_feature_relationships(self, save_path: Optional[str] = None) -> None:
        """
        Analyze relationships between key features and target.
        
        Args:
            save_path: Optional path to save the figure
        """
        features = ['Duration', 'Heart_Rate', 'Body_Temp', 'Weight']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            axes[idx].scatter(
                self.train_df[feature], 
                self.train_df['Calories'],
                alpha=0.3,
                s=10
            )
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Calories')
            axes[idx].set_title(f'Calories vs {feature}')
            axes[idx].grid(True, alpha=0.3)
            
            z = np.polyfit(self.train_df[feature], self.train_df['Calories'], 1)
            p = np.poly1d(z)
            axes[idx].plot(
                self.train_df[feature], 
                p(self.train_df[feature]), 
                "r--", 
                alpha=0.8, 
                linewidth=2
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_full_report(self, output_dir: str = "analysis_results") -> None:
        """
        Generate complete analysis report with all visualizations.
        
        Args:
            output_dir: Directory to save all analysis outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("="*60)
        print("CALORIE EXPENDITURE PREDICTION - DATA ANALYSIS REPORT")
        print("="*60)
        
        print("\n1. DATASET OVERVIEW")
        print("-" * 60)
        print(f"Training samples: {len(self.train_df)}")
        if self.test_df is not None:
            print(f"Test samples: {len(self.test_df)}")
        print(f"\nFeatures: {list(self.train_df.columns)}")
        
        print("\n2. BASIC STATISTICS")
        print("-" * 60)
        stats = self.get_basic_statistics()
        print(stats)
        
        print("\n3. TARGET DISTRIBUTION ANALYSIS")
        print("-" * 60)
        self.analyze_target_distribution(save_path=str(output_path / "target_distribution.png"))
        
        print("\n4. FEATURE DISTRIBUTIONS")
        print("-" * 60)
        self.plot_feature_distributions(save_path=str(output_path / "feature_distributions.png"))
        
        print("\n5. CORRELATION ANALYSIS")
        print("-" * 60)
        self.plot_correlation_matrix(save_path=str(output_path / "correlation_matrix.png"))
        
        print("\n6. GENDER-BASED ANALYSIS")
        print("-" * 60)
        self.analyze_gender_differences(save_path=str(output_path / "gender_analysis.png"))
        
        print("\n7. FEATURE RELATIONSHIPS")
        print("-" * 60)
        self.analyze_feature_relationships(save_path=str(output_path / "feature_relationships.png"))
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE - All visualizations saved to:", output_dir)
        print("="*60)
