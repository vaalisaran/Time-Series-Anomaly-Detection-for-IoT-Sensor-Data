
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineer:
    
    def __init__(self, data):
        
        self.data = data.copy()
        self.sensor_cols = [col for col in data.columns if 'sensor' in col]
        self.scaler = StandardScaler()
        self.feature_data = None
        
    def create_rolling_features(self, windows=[5, 10, 20]):
        
        logger.info("Creating rolling features...")
        
        feature_df = self.data.copy()
        
        for sensor in self.sensor_cols:
            for window in windows:
                # Rolling mean
                feature_df[f'{sensor}_rolling_mean_{window}'] = (
                    feature_df[sensor].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std
                feature_df[f'{sensor}_rolling_std_{window}'] = (
                    feature_df[sensor].rolling(window=window, min_periods=1).std()
                )
                
                # Rolling min/max
                feature_df[f'{sensor}_rolling_min_{window}'] = (
                    feature_df[sensor].rolling(window=window, min_periods=1).min()
                )
                
                feature_df[f'{sensor}_rolling_max_{window}'] = (
                    feature_df[sensor].rolling(window=window, min_periods=1).max()
                )
                
                # Rolling range
                feature_df[f'{sensor}_rolling_range_{window}'] = (
                    feature_df[f'{sensor}_rolling_max_{window}'] - 
                    feature_df[f'{sensor}_rolling_min_{window}']
                )
        
        self.feature_data = feature_df
        logger.info(f"Created rolling features. New shape: {feature_df.shape}")
        return feature_df
    
    def create_lag_features(self, lags=[1, 2, 5]):
        
        logger.info("Creating lag features...")
        
        if self.feature_data is None:
            self.feature_data = self.data.copy()
        
        for sensor in self.sensor_cols:
            for lag in lags:
                self.feature_data[f'{sensor}_lag_{lag}'] = (
                    self.feature_data[sensor].shift(lag)
                )
        
        logger.info(f"Created lag features. New shape: {self.feature_data.shape}")
        return self.feature_data
    
    def create_difference_features(self):
        logger.info("Creating difference features...")
        
        if self.feature_data is None:
            self.feature_data = self.data.copy()
        
        for sensor in self.sensor_cols:
            # First order difference
            self.feature_data[f'{sensor}_diff1'] = (
                self.feature_data[sensor].diff()
            )
            
            # Second order difference
            self.feature_data[f'{sensor}_diff2'] = (
                self.feature_data[sensor].diff().diff()
            )
        
        logger.info(f"Created difference features. New shape: {self.feature_data.shape}")
        return self.feature_data
    
    def create_statistical_features(self, window=20):
        
        logger.info("Creating statistical features...")
        
        if self.feature_data is None:
            self.feature_data = self.data.copy()
        
        for sensor in self.sensor_cols:
            # Rolling skewness
            self.feature_data[f'{sensor}_skew_{window}'] = (
                self.feature_data[sensor].rolling(window=window, min_periods=1).skew()
            )
            
            # Rolling kurtosis
            self.feature_data[f'{sensor}_kurt_{window}'] = (
                self.feature_data[sensor].rolling(window=window, min_periods=1).kurt()
            )
        
        logger.info(f"Created statistical features. New shape: {self.feature_data.shape}")
        return self.feature_data
    
    def create_interaction_features(self):
        logger.info("Creating interaction features...")
        
        if self.feature_data is None:
            self.feature_data = self.data.copy()
        
        if len(self.sensor_cols) >= 2:
            for i in range(len(self.sensor_cols)):
                for j in range(i+1, min(i+3, len(self.sensor_cols))):  # Limit interactions
                    sensor_i = self.sensor_cols[i]
                    sensor_j = self.sensor_cols[j]
                    
                    # Ratio
                    self.feature_data[f'{sensor_i}_{sensor_j}_ratio'] = (
                        self.feature_data[sensor_i] / (self.feature_data[sensor_j] + 1e-8)
                    )
                    
                    # Product
                    self.feature_data[f'{sensor_i}_{sensor_j}_product'] = (
                        self.feature_data[sensor_i] * self.feature_data[sensor_j]
                    )
        
        logger.info(f"Created interaction features. New shape: {self.feature_data.shape}")
        return self.feature_data
    
    def create_all_features(self):
        logger.info("Creating all features...")
        
        self.create_rolling_features(windows=[5, 10, 20])
        self.create_lag_features(lags=[1, 2, 5])
        self.create_difference_features()
        self.create_statistical_features(window=20)
        self.create_interaction_features()
        
        # Fill NaN values created by rolling/lag operations
        self.feature_data.fillna(method='bfill', inplace=True)
        self.feature_data.fillna(method='ffill', inplace=True)
        self.feature_data.fillna(0, inplace=True)
        
        logger.info(f"All features created. Final shape: {self.feature_data.shape}")
        return self.feature_data
    
    def scale_features(self, method='standard'):
        
        logger.info(f"Scaling features using {method} scaling...")
        
        if self.feature_data is None:
            raise ValueError("No features created. Call create_all_features() first.")
        
        # Get numeric columns only
        numeric_cols = self.feature_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['timestamp', 'file_number']]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        
        self.feature_data[numeric_cols] = self.scaler.fit_transform(
            self.feature_data[numeric_cols]
        )
        
        logger.info("Features scaled successfully")
        return self.feature_data, self.scaler
    
    def get_feature_names(self):
        """Get list of all feature names"""
        if self.feature_data is None:
            return []
        
        feature_cols = [col for col in self.feature_data.columns 
                       if col not in ['timestamp', 'file_number']]
        return feature_cols


if __name__ == "__main__":
    # Example usage
    from data_loader import NASABearingDataLoader
    
    loader = NASABearingDataLoader("nasa_bearing_dataset")
    data = loader.load_bearing_data('Bearing1_1')
    
    engineer = TimeSeriesFeatureEngineer(data)
    features = engineer.create_all_features()
    scaled_features, scaler = engineer.scale_features()
    
    print(f"Total features: {len(engineer.get_feature_names())}")