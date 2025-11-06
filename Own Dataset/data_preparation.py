#Load, clean, and perform exploratory data analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, filepath):
        """Initialize with data file path"""
        self.filepath = filepath
        self.df = None
        self.sensor_cols = None
        
    def load_data(self):
        """Load the sensor data"""
        print("Loading data...")
        self.df = pd.read_csv(self.filepath)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.sensor_cols = [col for col in self.df.columns if col.startswith('sensor_')]
        print(f"Data loaded: {self.df.shape}")
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\nChecking for missing values...")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        
        if missing.sum() > 0:
            self.df[self.sensor_cols] = self.df[self.sensor_cols].fillna(method='ffill')
            self.df[self.sensor_cols] = self.df[self.sensor_cols].fillna(method='bfill')
            print("Missing values handled using forward/backward fill")
        else:
            print("No missing values found")
            
        return self.df
    
    def detect_outliers_zscore(self, threshold=3):
        """Detect outliers using Z-score method"""
        print(f"\nDetecting outliers (Z-score > {threshold})...")
        outlier_counts = {}
        
        for col in self.sensor_cols:
            z_scores = np.abs(stats.zscore(self.df[col]))
            outliers = z_scores > threshold
            outlier_counts[col] = outliers.sum()
            
        print("Outlier counts per sensor:")
        for sensor, count in outlier_counts.items():
            print(f"  {sensor}: {count} outliers ({count/len(self.df)*100:.2f}%)")
            
        return outlier_counts
    
    def get_basic_statistics(self):
        """Get basic statistical summary"""
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        print(self.df[self.sensor_cols].describe())
        
        return self.df[self.sensor_cols].describe()
    
    def plot_time_series(self, save_path='plots/time_series.png'):
        """Plot time series for all sensors"""
        import os
        os.makedirs('plots', exist_ok=True)
        
        fig, axes = plt.subplots(len(self.sensor_cols), 1, figsize=(15, 10))
        if len(self.sensor_cols) == 1:
            axes = [axes]
            
        for idx, sensor in enumerate(self.sensor_cols):
            axes[idx].plot(self.df['timestamp'], self.df[sensor], linewidth=0.5, alpha=0.7)
            axes[idx].set_ylabel(sensor)
            axes[idx].set_title(f'{sensor} Time Series')
            axes[idx].grid(True, alpha=0.3)
            
        plt.xlabel('Timestamp')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTime series plot saved to {save_path}")
        plt.close()
        
    def plot_distributions(self, save_path='plots/distributions.png'):
        """Plot distribution of sensor values"""
        import os
        os.makedirs('plots', exist_ok=True)
        
        fig, axes = plt.subplots(2, (len(self.sensor_cols)+1)//2, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, sensor in enumerate(self.sensor_cols):
            axes[idx].hist(self.df[sensor], bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{sensor} Distribution')
            axes[idx].grid(True, alpha=0.3)
            
        # Hide extra subplots
        for idx in range(len(self.sensor_cols), len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
        plt.close()
        
    def plot_correlation_matrix(self, save_path='plots/correlation_matrix.png'):
        """Plot correlation matrix between sensors"""
        import os
        os.makedirs('plots', exist_ok=True)
        
        correlation_matrix = self.df[self.sensor_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Sensor Correlation Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {save_path}")
        plt.close()
        
        return correlation_matrix
    
    def plot_boxplots(self, save_path='plots/boxplots.png'):
        """Plot boxplots to visualize outliers"""
        import os
        os.makedirs('plots', exist_ok=True)
        
        fig, axes = plt.subplots(1, len(self.sensor_cols), figsize=(15, 5))
        if len(self.sensor_cols) == 1:
            axes = [axes]
            
        for idx, sensor in enumerate(self.sensor_cols):
            axes[idx].boxplot(self.df[sensor])
            axes[idx].set_ylabel('Value')
            axes[idx].set_title(f'{sensor}')
            axes[idx].grid(True, alpha=0.3)
            
        plt.suptitle('Sensor Value Distributions (Boxplots)', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Boxplots saved to {save_path}")
        plt.close()
        
    def run_eda(self):
        """Run complete exploratory data analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        self.load_data()
        self.handle_missing_values()
        self.get_basic_statistics()
        self.detect_outliers_zscore()
        
        # Generate visualizations
        self.plot_time_series()
        self.plot_distributions()
        self.plot_correlation_matrix()
        self.plot_boxplots()
        
        print("\n" + "="*60)
        print("EDA COMPLETE - All plots saved to 'plots/' directory")
        print("="*60)
        
        return self.df

if __name__ == "__main__":
    # Run EDA
    prep = DataPreparation('sensor_data.csv')
    df = prep.run_eda()