"""
File: visualization.py
Purpose: Visualization functions for anomaly detection
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class AnomalyVisualizer:
    """Visualization tools for anomaly detection"""
    
    @staticmethod
    def plot_sensor_data(data, sensor_cols, title="Sensor Readings Over Time", 
                        max_sensors=4):
        """
        Plot raw sensor data
        
        Args:
            data: DataFrame with sensor data
            sensor_cols: List of sensor column names
            title: Plot title
            max_sensors: Maximum sensors to plot
        """
        fig, axes = plt.subplots(min(len(sensor_cols), max_sensors), 1, 
                                figsize=(14, 3*min(len(sensor_cols), max_sensors)))
        
        if min(len(sensor_cols), max_sensors) == 1:
            axes = [axes]
        
        for idx, sensor in enumerate(sensor_cols[:max_sensors]):
            axes[idx].plot(data[sensor], linewidth=0.5, alpha=0.7)
            axes[idx].set_title(f'{sensor}', fontsize=10)
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=1.00)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_correlation_matrix(data, sensor_cols, title="Sensor Correlation Matrix"):
        """
        Plot correlation matrix of sensors
        
        Args:
            data: DataFrame with sensor data
            sensor_cols: List of sensor column names
            title: Plot title
        """
        correlation = data[sensor_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.2f', ax=ax)
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_distribution(data, sensor_cols, title="Sensor Value Distributions",
                         max_sensors=4):
        """
        Plot distribution of sensor values
        
        Args:
            data: DataFrame with sensor data
            sensor_cols: List of sensor column names
            title: Plot title
            max_sensors: Maximum sensors to plot
        """
        fig, axes = plt.subplots(1, min(len(sensor_cols), max_sensors),
                                figsize=(4*min(len(sensor_cols), max_sensors), 4))
        
        if min(len(sensor_cols), max_sensors) == 1:
            axes = [axes]
        
        for idx, sensor in enumerate(sensor_cols[:max_sensors]):
            axes[idx].hist(data[sensor], bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{sensor}')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_anomalies_on_timeseries(data, sensor_col, anomaly_indices, 
                                     title="Detected Anomalies"):
        """
        Plot time series with anomalies highlighted
        
        Args:
            data: DataFrame with sensor data
            sensor_col: Sensor column to plot
            anomaly_indices: Indices of detected anomalies
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot normal data
        ax.plot(data.index, data[sensor_col], label='Normal', 
               linewidth=0.8, alpha=0.7, color='blue')
        
        # Highlight anomalies
        if len(anomaly_indices) > 0:
            ax.scatter(anomaly_indices, data.iloc[anomaly_indices][sensor_col],
                      color='red', s=30, label='Anomaly', zorder=5, alpha=0.8)
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Sensor Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_multiple_sensors_with_anomalies(data, sensor_cols, anomaly_indices,
                                            title="Anomalies Across Sensors",
                                            max_sensors=4):
        """
        Plot multiple sensors with anomalies
        
        Args:
            data: DataFrame with sensor data
            sensor_cols: List of sensor columns
            anomaly_indices: Indices of detected anomalies
            title: Plot title
            max_sensors: Maximum sensors to plot
        """
        fig, axes = plt.subplots(min(len(sensor_cols), max_sensors), 1,
                                figsize=(14, 3*min(len(sensor_cols), max_sensors)))
        
        if min(len(sensor_cols), max_sensors) == 1:
            axes = [axes]
        
        for idx, sensor in enumerate(sensor_cols[:max_sensors]):
            # Plot normal data
            axes[idx].plot(data.index, data[sensor], linewidth=0.6, 
                          alpha=0.6, color='blue')
            
            # Highlight anomalies
            if len(anomaly_indices) > 0:
                axes[idx].scatter(anomaly_indices, 
                                data.iloc[anomaly_indices][sensor],
                                color='red', s=20, zorder=5, alpha=0.8)
            
            axes[idx].set_ylabel(sensor)
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Index')
        plt.suptitle(title, fontsize=14, y=1.00)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_anomaly_scores(scores, anomaly_indices, title="Anomaly Scores"):
        """
        Plot anomaly scores over time
        
        Args:
            scores: Array of anomaly scores
            anomaly_indices: Indices of detected anomalies
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot scores
        ax.plot(scores, linewidth=0.8, alpha=0.7, color='blue', label='Score')
        
        # Highlight anomalies
        if len(anomaly_indices) > 0:
            ax.scatter(anomaly_indices, scores[anomaly_indices],
                      color='red', s=30, label='Anomaly', zorder=5, alpha=0.8)
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Anomaly Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pca_with_anomalies(X, anomaly_indices, title="PCA Visualization"):
        """
        Plot PCA visualization with anomalies
        
        Args:
            X: Feature matrix
            anomaly_indices: Indices of detected anomalies
            title: Plot title
        """
        logger.info("Computing PCA...")
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot normal points
        normal_mask = np.ones(len(X), dtype=bool)
        normal_mask[anomaly_indices] = False
        
        ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
                  c='blue', s=20, alpha=0.5, label='Normal')
        
        # Plot anomalies
        if len(anomaly_indices) > 0:
            ax.scatter(X_pca[anomaly_indices, 0], X_pca[anomaly_indices, 1],
                      c='red', s=50, alpha=0.8, label='Anomaly', 
                      edgecolors='black', linewidth=1)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_reconstruction_error(errors, threshold, title="Reconstruction Error"):
        """
        Plot reconstruction error with threshold
        
        Args:
            errors: Array of reconstruction errors
            threshold: Anomaly threshold
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot errors
        ax.plot(errors, linewidth=0.8, alpha=0.7, color='blue', label='Error')
        
        # Plot threshold
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'Threshold ({threshold:.4f})')
        
        # Highlight anomalies
        anomaly_mask = errors > threshold
        if np.any(anomaly_mask):
            anomaly_indices = np.where(anomaly_mask)[0]
            ax.scatter(anomaly_indices, errors[anomaly_indices],
                      color='red', s=30, zorder=5, alpha=0.8)
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_model_comparison(metrics_dict, title="Model Comparison"):
        """
        Plot comparison of different models
        
        Args:
            metrics_dict: Dictionary of {model_name: metrics_dict}
            title: Plot title
        """
        models = list(metrics_dict.keys())
        n_anomalies = [metrics_dict[m]['n_anomalies'] for m in models]
        percentages = [metrics_dict[m]['anomaly_percentage'] for m in models]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Number of anomalies
        axes[0].bar(models, n_anomalies, color=['blue', 'green', 'orange'][:len(models)])
        axes[0].set_title('Number of Detected Anomalies')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Percentage of anomalies
        axes[1].bar(models, percentages, color=['blue', 'green', 'orange'][:len(models)])
        axes[1].set_title('Percentage of Anomalies')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(fig, filename, dpi=300):
        """Save figure to file"""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved to {filename}")


if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'sensor_1': np.random.randn(1000),
        'sensor_2': np.random.randn(1000)
    })
    
    visualizer = AnomalyVisualizer()
    fig = visualizer.plot_sensor_data(data, ['sensor_1', 'sensor_2'])
    plt.show()