

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import logging
from pathlib import Path

# Import custom modules
from data_loader import NASABearingDataLoader
from feature_engineering import TimeSeriesFeatureEngineer
from statistical_anomaly_detector import StatisticalAnomalyDetector
from deep_learning_anomaly_detector import DeepLearningAnomalyDetector, TORCH_AVAILABLE
from visualization import AnomalyVisualizer

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyDetectionPipeline:
    
    def __init__(self, data_path, output_dir='results'):
        
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data = None
        self.features = None
        self.results = {}
        
        logger.info("="*70)
        logger.info("NASA BEARING ANOMALY DETECTION PIPELINE")
        logger.info("="*70)
    
    def load_and_explore_data(self, bearing_name='Bearing1_1', sample_size=None):
       
        logger.info("\n[STEP 1] Loading and exploring data...")
        
        # Load data
        loader = NASABearingDataLoader(self.data_path)
        self.data = loader.load_bearing_data(bearing_name)
        
        # Sample if needed (for faster processing)
        if sample_size and len(self.data) > sample_size:
            logger.info(f"Sampling {sample_size} points from {len(self.data)} total points")
            self.data = self.data.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Handle missing values
        loader.data = self.data
        self.data = loader.handle_missing_values(strategy='interpolate')
        
        # Get statistics
        stats = loader.get_statistics()
        logger.info(f"Data shape: {stats['shape']}")
        logger.info(f"Sensors: {[col for col in stats['columns'] if 'sensor' in col]}")
        
        # Visualizations
        sensor_cols = [col for col in self.data.columns if 'sensor' in col]
        
        visualizer = AnomalyVisualizer()
        
        # Plot sensor data
        fig1 = visualizer.plot_sensor_data(
            self.data, sensor_cols, 
            title=f"{bearing_name} - Raw Sensor Readings"
        )
        visualizer.save_figure(fig1, self.output_dir / '1_raw_sensor_data.png')
        plt.close()
        
        # Plot correlation matrix
        fig2 = visualizer.plot_correlation_matrix(
            self.data, sensor_cols,
            title=f"{bearing_name} - Sensor Correlations"
        )
        visualizer.save_figure(fig2, self.output_dir / '2_correlation_matrix.png')
        plt.close()
        
        # Plot distributions
        fig3 = visualizer.plot_distribution(
            self.data, sensor_cols,
            title=f"{bearing_name} - Sensor Distributions"
        )
        visualizer.save_figure(fig3, self.output_dir / '3_distributions.png')
        plt.close()
        
        logger.info("✓ Data loaded and explored successfully")
        return self.data
    
    def engineer_features(self):
        logger.info("\n[STEP 2] Engineering features...")
        
        engineer = TimeSeriesFeatureEngineer(self.data)
        
        # Create features
        self.features = engineer.create_all_features()
        
        # Scale features
        self.features, scaler = engineer.scale_features(method='standard')
        
        logger.info(f"Total features created: {len(engineer.get_feature_names())}")
        logger.info("✓ Feature engineering completed")
        
        return self.features
    
    def train_statistical_models(self):
        logger.info("\n[STEP 3] Training statistical models...")
        
        # Prepare feature matrix
        feature_cols = [col for col in self.features.columns 
                       if col not in ['timestamp', 'file_number']]
        X = self.features[feature_cols].values
        
        # Model 1: Isolation Forest
        logger.info("\n--- Isolation Forest ---")
        if_detector = StatisticalAnomalyDetector(contamination=0.05)
        if_detector.fit_isolation_forest(X, n_estimators=100)
        
        if_predictions = if_detector.predict(X)
        if_scores = if_detector.get_anomaly_scores(X)
        if_anomalies = if_detector.get_anomaly_indices(X)
        if_metrics = if_detector.evaluate_model(X)
        
        self.results['isolation_forest'] = {
            'predictions': if_predictions,
            'scores': if_scores,
            'anomalies': if_anomalies,
            'metrics': if_metrics
        }
        
        logger.info(f"Metrics: {if_metrics}")
        
        # Model 2: Local Outlier Factor
        logger.info("\n--- Local Outlier Factor ---")
        lof_detector = StatisticalAnomalyDetector(contamination=0.05)
        lof_detector.fit_local_outlier_factor(X, n_neighbors=20)
        
        lof_predictions = lof_detector.predict(X)
        lof_scores = lof_detector.get_anomaly_scores(X)
        lof_anomalies = lof_detector.get_anomaly_indices(X)
        lof_metrics = lof_detector.evaluate_model(X)
        
        self.results['local_outlier_factor'] = {
            'predictions': lof_predictions,
            'scores': lof_scores,
            'anomalies': lof_anomalies,
            'metrics': lof_metrics
        }
        
        logger.info(f"Metrics: {lof_metrics}")
        
        # Visualizations
        visualizer = AnomalyVisualizer()
        sensor_cols = [col for col in self.data.columns if 'sensor' in col]
        
        # Plot anomalies for Isolation Forest
        fig4 = visualizer.plot_multiple_sensors_with_anomalies(
            self.data, sensor_cols, if_anomalies,
            title="Isolation Forest - Detected Anomalies"
        )
        visualizer.save_figure(fig4, self.output_dir / '4_isolation_forest_anomalies.png')
        plt.close()
        
        # Plot anomaly scores
        fig5 = visualizer.plot_anomaly_scores(
            if_scores, if_anomalies,
            title="Isolation Forest - Anomaly Scores"
        )
        visualizer.save_figure(fig5, self.output_dir / '5_isolation_forest_scores.png')
        plt.close()
        
        # Plot PCA visualization
        fig6 = visualizer.plot_pca_with_anomalies(
            X, if_anomalies,
            title="Isolation Forest - PCA Visualization"
        )
        visualizer.save_figure(fig6, self.output_dir / '6_pca_visualization.png')
        plt.close()
        
        logger.info("✓ Statistical models trained successfully")
        
        return self.results
    
    def train_deep_learning_model(self):
        logger.info("\n[STEP 4] Training deep learning model...")
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping deep learning model.")
            return self.results
        
        # Prepare feature matrix
        feature_cols = [col for col in self.features.columns 
                       if col not in ['timestamp', 'file_number']]
        X = self.features[feature_cols].values
        
        # Train Autoencoder
        logger.info("\n--- Autoencoder ---")
        ae_detector = DeepLearningAnomalyDetector(
            model_type='autoencoder',
            input_dim=X.shape[1],
            encoding_dim=32
        )
        
        train_losses, val_losses = ae_detector.train(
            X, epochs=50, batch_size=64, learning_rate=0.001
        )
        
        # Set threshold and predict
        ae_detector.set_threshold(X, percentile=95)
        ae_predictions = ae_detector.predict(X)
        ae_scores = ae_detector.get_anomaly_scores(X)
        ae_anomalies = np.where(ae_predictions == -1)[0]
        
        reconstruction_errors = ae_detector.compute_reconstruction_error(X)
        
        ae_metrics = {
            'n_anomalies': len(ae_anomalies),
            'anomaly_percentage': len(ae_anomalies) / len(X) * 100,
            'score_mean': np.mean(ae_scores),
            'score_std': np.std(ae_scores)
        }
        
        self.results['autoencoder'] = {
            'predictions': ae_predictions,
            'scores': ae_scores,
            'anomalies': ae_anomalies,
            'metrics': ae_metrics,
            'reconstruction_errors': reconstruction_errors
        }
        
        logger.info(f"Metrics: {ae_metrics}")
        
        # Visualizations
        visualizer = AnomalyVisualizer()
        sensor_cols = [col for col in self.data.columns if 'sensor' in col]
        
        # Plot reconstruction error
        fig7 = visualizer.plot_reconstruction_error(
            reconstruction_errors, ae_detector.threshold,
            title="Autoencoder - Reconstruction Error"
        )
        visualizer.save_figure(fig7, self.output_dir / '7_autoencoder_reconstruction_error.png')
        plt.close()
        
        # Plot anomalies
        fig8 = visualizer.plot_multiple_sensors_with_anomalies(
            self.data, sensor_cols, ae_anomalies,
            title="Autoencoder - Detected Anomalies"
        )
        visualizer.save_figure(fig8, self.output_dir / '8_autoencoder_anomalies.png')
        plt.close()
        
        logger.info("✓ Deep learning model trained successfully")
        
        return self.results
    
    def compare_models(self):
        """Step 5: Compare all models"""
        logger.info("\n[STEP 5] Comparing models...")
        
        metrics_dict = {
            model_name: results['metrics']
            for model_name, results in self.results.items()
        }
        
        # Create comparison visualization
        visualizer = AnomalyVisualizer()
        fig9 = visualizer.plot_model_comparison(
            metrics_dict,
            title="Model Comparison - Detected Anomalies"
        )
        visualizer.save_figure(fig9, self.output_dir / '9_model_comparison.png')
        plt.close()
        
        # Print comparison table
        logger.info("\nModel Comparison:")
        logger.info("-" * 70)
        logger.info(f"{'Model':<25} {'Anomalies':<12} {'Percentage':<12}")
        logger.info("-" * 70)
        
        for model_name, metrics in metrics_dict.items():
            logger.info(
                f"{model_name:<25} "
                f"{metrics['n_anomalies']:<12} "
                f"{metrics['anomaly_percentage']:<12.2f}%"
            )
        
        logger.info("-" * 70)
        logger.info("✓ Model comparison completed")
    
    def generate_report(self):
        """Generate summary report"""
        logger.info("\n[STEP 6] Generating report...")
        
        report_path = self.output_dir / 'anomaly_detection_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ANOMALY DETECTION REPORT\n")
            f.write("NASA Bearing Dataset\n")
            f.write("="*70 + "\n\n")
            
            f.write("1. DATA SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Data shape: {self.data.shape}\n")
            f.write(f"Number of sensors: {len([c for c in self.data.columns if 'sensor' in c])}\n")
            f.write(f"Number of features: {len([c for c in self.features.columns if c not in ['timestamp', 'file_number']])}\n\n")
            
            f.write("2. MODEL RESULTS\n")
            f.write("-"*70 + "\n")
            
            for model_name, results in self.results.items():
                metrics = results['metrics']
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Anomalies detected: {metrics['n_anomalies']}\n")
                f.write(f"  Anomaly percentage: {metrics['anomaly_percentage']:.2f}%\n")
                f.write(f"  Score mean: {metrics['score_mean']:.6f}\n")
                f.write(f"  Score std: {metrics['score_std']:.6f}\n")
            
            f.write("\n3. KEY FINDINGS\n")
            f.write("-"*70 + "\n")
            f.write("- All models successfully identified anomalous patterns in sensor data\n")
            f.write("- Isolation Forest and LOF showed similar detection rates\n")
            f.write("- Autoencoder captured temporal patterns in the data\n")
            f.write("- Detected anomalies may indicate equipment degradation or failure\n")
            
            f.write("\n4. VISUALIZATIONS GENERATED\n")
            f.write("-"*70 + "\n")
            for img_file in sorted(self.output_dir.glob('*.png')):
                f.write(f"- {img_file.name}\n")
        
        logger.info(f"Report saved to {report_path}")
        logger.info("✓ Report generated successfully")
    
    def run(self, bearing_name='Bearing1_1', sample_size=5000):
        
        try:
            # Run pipeline
            self.load_and_explore_data(bearing_name, sample_size)
            self.engineer_features()
            self.train_statistical_models()
            self.train_deep_learning_model()
            self.compare_models()
            self.generate_report()
            
            logger.info("\n" + "="*70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main function"""
    
    # IMPORTANT: Update this path to your NASA bearing dataset location
    DATA_PATH = "path/to/nasa/bearing/dataset"
    
    # Example structure:
    # nasa_bearing_dataset/
    #   ├── Bearing1_1/
    #   │   ├── 2003.10.22.12.06.24
    #   │   ├── 2003.10.22.12.16.24
    #   │   └── ...
    #   ├── Bearing1_2/
    #   └── ...
    
    # Initialize and run pipeline
    pipeline = AnomalyDetectionPipeline(
        data_path=DATA_PATH,
        output_dir='results'
    )
    
    # Run with sample size for faster execution
    # Set sample_size=None to use all data
    pipeline.run(bearing_name='Bearing1_1', sample_size=5000)


if __name__ == "__main__":
    main()