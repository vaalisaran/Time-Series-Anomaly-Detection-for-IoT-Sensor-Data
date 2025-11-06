# File: main.py
# Purpose: Main execution script - runs the complete anomaly detection pipeline

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    dirs = ['plots', 'models', 'results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Created necessary directories")

def run_pipeline(n_samples=10000, n_sensors=5, anomaly_ratio=0.05):
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("STARTING ANOMALY DETECTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Samples: {n_samples}")
    logger.info(f"  Sensors: {n_sensors}")
    logger.info(f"  Anomaly Ratio: {anomaly_ratio*100:.1f}%")
    
    try:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Generating Synthetic Data")
        logger.info("="*80)
        from data_generator import SensorDataGenerator
        generator = SensorDataGenerator(
            n_samples=n_samples,
            n_sensors=n_sensors,
            anomaly_ratio=anomaly_ratio
        )
        df = generator.generate('sensor_data.csv')
        logger.info(f"✓ Data generation complete: {df.shape}")
        
        # Step 2: Data preparation and EDA
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Data Preparation & EDA")
        logger.info("="*80)
        from data_preparation import DataPreparation
        prep = DataPreparation('sensor_data.csv')
        df = prep.run_eda()
        logger.info("✓ EDA complete")
        
        # Step 3: Feature engineering
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Feature Engineering")
        logger.info("="*80)
        from feature_engineering import FeatureEngineering
        fe = FeatureEngineering(df)
        df_features, feature_cols = fe.create_all_features(normalize=True)
        df_features.to_csv('sensor_data_features.csv', index=False)
        logger.info(f"✓ Feature engineering complete: {len(feature_cols)} features created")
        
        # Step 4: Statistical anomaly detection
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Statistical Anomaly Detection")
        logger.info("="*80)
        from statistical_anomaly_detector import StatisticalAnomalyDetector
        
        # Prepare features
        exclude_cols = ['timestamp', 'is_anomaly']
        X = df_features[[col for col in df_features.columns if col not in exclude_cols]].values
        
        stat_detector = StatisticalAnomalyDetector(contamination=anomaly_ratio)
        stat_predictions, stat_scores = stat_detector.run_all_methods(X)
        
        # Add predictions to dataframe
        for method, pred in stat_predictions.items():
            df_features[f'pred_{method}'] = pred
        
        df_features.to_csv('results_statistical.csv', index=False)
        logger.info("✓ Statistical detection complete")
        
        # Step 5: Deep learning anomaly detection
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Deep Learning Anomaly Detection")
        logger.info("="*80)
        from deep_learning_anomaly_detector import DeepLearningAnomalyDetector
        from sklearn.model_selection import train_test_split
        
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        
        dl_detector = DeepLearningAnomalyDetector(contamination=anomaly_ratio)
        
        # Train Autoencoder
        logger.info("\nTraining Autoencoder...")
        autoencoder, ae_history = dl_detector.train_autoencoder(
            X_train, X_val, epochs=50, batch_size=128
        )
        ae_pred, ae_errors = dl_detector.predict_autoencoder(X, autoencoder, percentile=95)
        
        # Train LSTM Autoencoder
        logger.info("\nTraining LSTM Autoencoder...")
        lstm_ae, lstm_history = dl_detector.train_lstm_autoencoder(
            X_train, X_val, timesteps=10, epochs=30
        )
        lstm_pred, lstm_errors = dl_detector.predict_lstm_autoencoder(
            X, lstm_ae, timesteps=10, percentile=95
        )
        
        # Save models
        dl_detector.save_models()
        
        # Add predictions
        df_features['pred_autoencoder'] = ae_pred
        df_features['pred_lstm_autoencoder'] = lstm_pred
        df_features['error_autoencoder'] = ae_errors
        df_features['error_lstm_autoencoder'] = lstm_errors
        
        df_features.to_csv('results_deep_learning.csv', index=False)
        logger.info("✓ Deep learning detection complete")
        
        # Step 6: Model evaluation
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Model Evaluation & Comparison")
        logger.info("="*80)
        from model_evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator(df_features, true_label_col='is_anomaly')
        results = evaluator.run_full_evaluation()
        
        logger.info("✓ Evaluation complete")
        
        # Save final results
        results.to_csv('results/model_comparison.csv', index=False)
        df_features.to_csv('results/final_results.csv', index=False)
        
        # Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Data samples processed: {len(df_features):,}")
        logger.info(f"Features created: {len(feature_cols)}")
        logger.info(f"Models trained: {len(stat_predictions) + 2}")
        logger.info("\nTop 3 Models by F1-Score:")
        for idx, row in results.head(3).iterrows():
            logger.info(f"  {idx+1}. {row['method']}: F1={row['f1_score']:.4f}, "
                       f"Precision={row['precision']:.4f}, Recall={row['recall']:.4f}")
        
        logger.info("\nOutput Files Generated:")
        logger.info("  Data Files:")
        logger.info("    - sensor_data.csv (raw data)")
        logger.info("    - sensor_data_features.csv (with features)")
        logger.info("    - results/final_results.csv (complete results)")
        logger.info("  Model Files:")
        logger.info("    - models/statistical_models.pkl")
        logger.info("    - models/autoencoder.h5")
        logger.info("    - models/lstm_autoencoder.h5")
        logger.info("    - models/scaler.pkl")
        logger.info("  Evaluation:")
        logger.info("    - results/model_comparison.csv")
        logger.info("    - evaluation_report.txt")
        logger.info("  Plots:")
        logger.info("    - plots/time_series.png")
        logger.info("    - plots/distributions.png")
        logger.info("    - plots/correlation_matrix.png")
        logger.info("    - plots/confusion_matrices.png")
        logger.info("    - plots/roc_curves.png")
        logger.info("    - plots/pr_curves.png")
        logger.info("    - plots/anomaly_timeline.png")
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        return df_features, results
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Create necessary directories
    create_directories()
    
    # Run the complete pipeline
    try:
        df_final, model_results = run_pipeline(
            n_samples=10000,
            n_sensors=5,
            anomaly_ratio=0.05
        )
        
        print("\n" + "="*80)
        print("SUCCESS! Check the following:")
        print("  - 'plots/' directory for visualizations")
        print("  - 'models/' directory for saved models")
        print("  - 'results/' directory for final outputs")
        print("  - 'evaluation_report.txt' for detailed comparison")
        print("  - 'pipeline.log' for execution logs")
        print("="*80)
        
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed: {str(e)}")
        sys.exit(1)