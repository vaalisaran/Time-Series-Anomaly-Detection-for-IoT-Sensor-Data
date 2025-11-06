
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import pickle
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnomalyDetector:
    def __init__(self, contamination=0.05):
        
        self.contamination = contamination
        self.models = {}
        self.predictions = {}
        self.scores = {}
        
    def isolation_forest(self, X, n_estimators=100, random_state=42):
        
        print("\n" + "="*60)
        print("ISOLATION FOREST")
        print("="*60)
        print(f"Training with {n_estimators} estimators...")
        print(f"Expected contamination: {self.contamination*100:.1f}%")
        
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=self.contamination,
            random_state=random_state,
            n_jobs=-1,
            max_samples='auto'
        )
        
        # Fit and predict
        predictions = model.fit_predict(X)
        scores = model.score_samples(X)
        
        # Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0
        predictions = (predictions == -1).astype(int)
        
        self.models['isolation_forest'] = model
        self.predictions['isolation_forest'] = predictions
        self.scores['isolation_forest'] = scores
        
        n_anomalies = predictions.sum()
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(predictions)*100:.2f}%)")
        print("="*60)
        
        return predictions, scores
    
    def local_outlier_factor(self, X, n_neighbors=20):
        
        print("\n" + "="*60)
        print("LOCAL OUTLIER FACTOR")
        print("="*60)
        print(f"Using {n_neighbors} neighbors...")
        print(f"Expected contamination: {self.contamination*100:.1f}%")
        
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1
        )
        
        # Fit and predict
        predictions = model.fit_predict(X)
        scores = model.negative_outlier_factor_
        
        # Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0
        predictions = (predictions == -1).astype(int)
        
        self.models['lof'] = model
        self.predictions['lof'] = predictions
        self.scores['lof'] = scores
        
        n_anomalies = predictions.sum()
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(predictions)*100:.2f}%)")
        print("="*60)
        
        return predictions, scores
    
    def elliptic_envelope(self, X, random_state=42):
        
        print("\n" + "="*60)
        print("ELLIPTIC ENVELOPE")
        print("="*60)
        print(f"Expected contamination: {self.contamination*100:.1f}%")
        
        model = EllipticEnvelope(
            contamination=self.contamination,
            random_state=random_state
        )
        
        # Fit and predict
        predictions = model.fit_predict(X)
        scores = model.score_samples(X)
        
        # Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0
        predictions = (predictions == -1).astype(int)
        
        self.models['elliptic_envelope'] = model
        self.predictions['elliptic_envelope'] = predictions
        self.scores['elliptic_envelope'] = scores
        
        n_anomalies = predictions.sum()
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(predictions)*100:.2f}%)")
        print("="*60)
        
        return predictions, scores
    
    def z_score_method(self, X, threshold=3):
        
        print("\n" + "="*60)
        print("Z-SCORE METHOD")
        print("="*60)
        print(f"Using threshold: {threshold} standard deviations")
        
        # Calculate z-scores
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        z_scores = np.abs((X - mean) / (std + 1e-6))
        
        # Take maximum z-score across features
        max_z_scores = np.max(z_scores, axis=1)
        
        # Predict anomalies
        predictions = (max_z_scores > threshold).astype(int)
        
        self.predictions['z_score'] = predictions
        self.scores['z_score'] = -max_z_scores  # Negative for consistency
        
        n_anomalies = predictions.sum()
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(predictions)*100:.2f}%)")
        print("="*60)
        
        return predictions, -max_z_scores
    
    def ensemble_predict(self, min_votes=2):
        
        print("\n" + "="*60)
        print("ENSEMBLE PREDICTION")
        print("="*60)
        print(f"Using {len(self.predictions)} models")
        print(f"Minimum votes required: {min_votes}")
        
        # Stack all predictions
        all_predictions = np.column_stack(list(self.predictions.values()))
        
        # Count votes
        votes = np.sum(all_predictions, axis=1)
        
        # Predict anomaly if at least min_votes models agree
        ensemble_pred = (votes >= min_votes).astype(int)
        
        self.predictions['ensemble'] = ensemble_pred
        
        n_anomalies = ensemble_pred.sum()
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(ensemble_pred)*100:.2f}%)")
        print("="*60)
        
        return ensemble_pred
    
    def save_models(self, path='models/statistical_models.pkl'):
        """Save trained models"""
        import os
        os.makedirs('models', exist_ok=True)
        
        save_dict = {
            'models': self.models,
            'contamination': self.contamination
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"\nModels saved to {path}")
        
    def run_all_methods(self, X):
        """Run all statistical methods"""
        print("\n" + "="*60)
        print("STATISTICAL ANOMALY DETECTION")
        print("="*60)
        
        # Run all methods
        self.isolation_forest(X)
        self.local_outlier_factor(X)
        self.elliptic_envelope(X)
        self.z_score_method(X, threshold=3)
        
        # Ensemble prediction
        self.ensemble_predict(min_votes=2)
        
        self.save_models()
        
        return self.predictions, self.scores

if __name__ == "__main__":
    # Load featured data
    df = pd.read_csv('sensor_data_features.csv')
    
    # Prepare features
    exclude_cols = ['timestamp', 'is_anomaly']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    
    # Run statistical detection
    detector = StatisticalAnomalyDetector(contamination=0.05)
    predictions, scores = detector.run_all_methods(X)
    
    # Add predictions to dataframe
    for method, pred in predictions.items():
        df[f'pred_{method}'] = pred
        
    # Save results
    df.to_csv('results_statistical.csv', index=False)
    print("\nResults saved to 'results_statistical.csv'")