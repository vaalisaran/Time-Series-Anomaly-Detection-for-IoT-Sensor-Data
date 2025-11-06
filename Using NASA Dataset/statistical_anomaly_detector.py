"""
File: statistical_anomaly_detector.py
Purpose: Statistical anomaly detection models (Isolation Forest, LOF)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnomalyDetector:
    """Statistical methods for anomaly detection"""
    
    def __init__(self, contamination=0.1):
        """
        Initialize detector
        
        Args:
            contamination: Expected proportion of outliers in the dataset
        """
        self.contamination = contamination
        self.model = None
        self.model_type = None
        self.anomaly_scores = None
        
    def fit_isolation_forest(self, X, n_estimators=100, max_samples='auto', 
                            random_state=42):
        """
        Fit Isolation Forest model
        
        Intuition: Isolation Forest isolates anomalies by randomly selecting 
        a feature and then randomly selecting a split value. Anomalies are 
        easier to isolate (require fewer splits) than normal points.
        
        Args:
            X: Feature matrix
            n_estimators: Number of trees
            max_samples: Number of samples to draw for each tree
            random_state: Random seed
        """
        logger.info("Fitting Isolation Forest...")
        logger.info(f"Parameters: n_estimators={n_estimators}, contamination={self.contamination}")
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=self.contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X)
        self.model_type = 'isolation_forest'
        
        logger.info("Isolation Forest trained successfully")
        return self
    
    def fit_local_outlier_factor(self, X, n_neighbors=20, metric='minkowski'):
        """
        Fit Local Outlier Factor model
        
        Intuition: LOF measures local deviation of density of a sample with 
        respect to its neighbors. Points that have a substantially lower 
        density than their neighbors are considered outliers.
        
        Args:
            X: Feature matrix
            n_neighbors: Number of neighbors to use
            metric: Distance metric
        """
        logger.info("Fitting Local Outlier Factor...")
        logger.info(f"Parameters: n_neighbors={n_neighbors}, contamination={self.contamination}")
        
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            metric=metric,
            novelty=True,
            n_jobs=-1
        )
        
        self.model.fit(X)
        self.model_type = 'local_outlier_factor'
        
        logger.info("LOF trained successfully")
        return self
    
    def fit_elliptic_envelope(self, X, support_fraction=None):
        """
        Fit Elliptic Envelope (Gaussian) model
        
        Intuition: Assumes data comes from a known distribution (Gaussian).
        Fits an ellipse to the central data points, and points outside 
        the ellipse are considered anomalies.
        
        Args:
            X: Feature matrix
            support_fraction: Proportion of points to include in support
        """
        logger.info("Fitting Elliptic Envelope...")
        
        self.model = EllipticEnvelope(
            contamination=self.contamination,
            support_fraction=support_fraction,
            random_state=42
        )
        
        self.model.fit(X)
        self.model_type = 'elliptic_envelope'
        
        logger.info("Elliptic Envelope trained successfully")
        return self
    
    def predict(self, X):
        """
        Predict anomalies
        
        Args:
            X: Feature matrix
            
        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit_* method first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_anomaly_scores(self, X):
        """
        Get anomaly scores
        
        Args:
            X: Feature matrix
            
        Returns:
            scores: Anomaly scores (lower means more anomalous)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit_* method first.")
        
        if self.model_type == 'isolation_forest':
            # For Isolation Forest, decision_function returns the anomaly score
            scores = self.model.decision_function(X)
        elif self.model_type == 'local_outlier_factor':
            scores = self.model.decision_function(X)
        elif self.model_type == 'elliptic_envelope':
            scores = self.model.decision_function(X)
        
        self.anomaly_scores = scores
        return scores
    
    def get_anomaly_indices(self, X):
        """
        Get indices of detected anomalies
        
        Args:
            X: Feature matrix
            
        Returns:
            anomaly_indices: Indices where anomalies were detected
        """
        predictions = self.predict(X)
        anomaly_indices = np.where(predictions == -1)[0]
        
        logger.info(f"Detected {len(anomaly_indices)} anomalies ({len(anomaly_indices)/len(X)*100:.2f}%)")
        return anomaly_indices
    
    def evaluate_model(self, X, y_true=None):
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y_true: True labels (if available)
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        anomaly_scores = self.get_anomaly_scores(X)
        
        metrics = {
            'n_anomalies': np.sum(predictions == -1),
            'anomaly_percentage': np.sum(predictions == -1) / len(predictions) * 100,
            'score_mean': np.mean(anomaly_scores),
            'score_std': np.std(anomaly_scores),
            'score_min': np.min(anomaly_scores),
            'score_max': np.max(anomaly_scores)
        }
        
        if y_true is not None:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            # Convert predictions to binary (0 for normal, 1 for anomaly)
            y_pred = (predictions == -1).astype(int)
            
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1_score'] = f1_score(y_true, y_pred)
        
        return metrics


class ZScoreAnomalyDetector:
    """Z-score based anomaly detection"""
    
    def __init__(self, threshold=3.0):
        """
        Initialize Z-score detector
        
        Args:
            threshold: Z-score threshold (typically 3.0)
        """
        self.threshold = threshold
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """Fit the model by computing mean and std"""
        logger.info(f"Fitting Z-score model with threshold={self.threshold}")
        
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        return self
    
    def predict(self, X):
        """Predict anomalies based on Z-score"""
        if self.mean is None or self.std is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        z_scores = np.abs((X - self.mean) / (self.std + 1e-8))
        
        # A point is anomalous if any feature exceeds threshold
        predictions = np.where(np.any(z_scores > self.threshold, axis=1), -1, 1)
        
        return predictions
    
    def get_anomaly_scores(self, X):
        """Get maximum Z-score across features"""
        if self.mean is None or self.std is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        z_scores = np.abs((X - self.mean) / (self.std + 1e-8))
        max_z_scores = np.max(z_scores, axis=1)
        
        # Convert to negative scores (lower is more anomalous)
        return -max_z_scores


if __name__ == "__main__":
    # Example usage
    X = np.random.randn(1000, 10)
    
    # Isolation Forest
    detector_if = StatisticalAnomalyDetector(contamination=0.1)
    detector_if.fit_isolation_forest(X)
    predictions = detector_if.predict(X)
    scores = detector_if.get_anomaly_scores(X)
    
    print(detector_if.evaluate_model(X))