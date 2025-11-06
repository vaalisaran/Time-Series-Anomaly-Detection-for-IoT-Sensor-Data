# File: feature_engineering.py
# Purpose: Create meaningful features from raw time series data

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle

class FeatureEngineering:
    def __init__(self, df):
        
        self.df = df.copy()
        self.sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        self.scaler = None
        self.feature_cols = []
        
    def create_rolling_features(self, windows=[5, 10, 20]):
        
        print(f"\nCreating rolling features with windows: {windows}")
        
        for sensor in self.sensor_cols:
            for window in windows:
                # Rolling mean
                self.df[f'{sensor}_rolling_mean_{window}'] = (
                    self.df[sensor].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling standard deviation
                self.df[f'{sensor}_rolling_std_{window}'] = (
                    self.df[sensor].rolling(window=window, min_periods=1).std()
                )
                
                # Rolling min and max
                self.df[f'{sensor}_rolling_min_{window}'] = (
                    self.df[sensor].rolling(window=window, min_periods=1).min()
                )
                self.df[f'{sensor}_rolling_max_{window}'] = (
                    self.df[sensor].rolling(window=window, min_periods=1).max()
                )
                
        print(f"Created rolling features. New shape: {self.df.shape}")
        
    def create_lag_features(self, lags=[1, 2, 3, 5, 10]):
        
        print(f"\nCreating lag features with lags: {lags}")
        
        for sensor in self.sensor_cols:
            for lag in lags:
                self.df[f'{sensor}_lag_{lag}'] = self.df[sensor].shift(lag)
                
        # Fill NaN values created by lagging
        self.df.fillna(method='bfill', inplace=True)
        
        print(f"Created lag features. New shape: {self.df.shape}")
        
    def create_difference_features(self):
        print("\nCreating difference features...")
        
        for sensor in self.sensor_cols:
            # First order difference
            self.df[f'{sensor}_diff_1'] = self.df[sensor].diff()
            
            # Second order difference
            self.df[f'{sensor}_diff_2'] = self.df[sensor].diff().diff()
            
        # Fill NaN values
        self.df.fillna(method='bfill', inplace=True)
        
        print(f"Created difference features. New shape: {self.df.shape}")
        
    def create_rate_of_change(self, periods=[5, 10]):
        
        print(f"\nCreating rate of change features with periods: {periods}")
        
        for sensor in self.sensor_cols:
            for period in periods:
                self.df[f'{sensor}_roc_{period}'] = (
                    self.df[sensor].pct_change(periods=period) * 100
                )
                
        # Fill NaN and inf values
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.fillna(method='bfill', inplace=True)
        
        print(f"Created rate of change features. New shape: {self.df.shape}")
        
    def create_statistical_features(self, window=20):
        
        print(f"\nCreating statistical features with window: {window}")
        
        for sensor in self.sensor_cols:
            # Rolling skewness
            self.df[f'{sensor}_skew_{window}'] = (
                self.df[sensor].rolling(window=window, min_periods=1).skew()
            )
            
            # Rolling kurtosis
            self.df[f'{sensor}_kurt_{window}'] = (
                self.df[sensor].rolling(window=window, min_periods=1).kurt()
            )
            
        # Fill NaN values
        self.df.fillna(method='bfill', inplace=True)
        
        print(f"Created statistical features. New shape: {self.df.shape}")
        
    def create_cross_sensor_features(self):
        print("\nCreating cross-sensor features...")
        
        if len(self.sensor_cols) >= 2:
            # Ratio between sensors
            for i in range(len(self.sensor_cols)):
                for j in range(i+1, len(self.sensor_cols)):
                    sensor1 = self.sensor_cols[i]
                    sensor2 = self.sensor_cols[j]
                    
                    # Ratio
                    self.df[f'{sensor1}_{sensor2}_ratio'] = (
                        self.df[sensor1] / (self.df[sensor2] + 1e-6)
                    )
                    
                    # Difference
                    self.df[f'{sensor1}_{sensor2}_diff'] = (
                        self.df[sensor1] - self.df[sensor2]
                    )
                    
            # Mean and std across all sensors
            self.df['all_sensors_mean'] = self.df[self.sensor_cols].mean(axis=1)
            self.df['all_sensors_std'] = self.df[self.sensor_cols].std(axis=1)
            
        print(f"Created cross-sensor features. New shape: {self.df.shape}")
        
    def create_time_based_features(self):
        print("\nCreating time-based features...")
        
        # Hour of day
        self.df['hour'] = self.df['timestamp'].dt.hour
        
        # Day of week
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        
        # Is weekend
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour (to capture circular nature)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        
        # Cyclical encoding for day of week
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        
        print(f"Created time-based features. New shape: {self.df.shape}")
        
    def normalize_features(self, method='robust'):
        
        print(f"\nNormalizing features using {method} scaler...")
        
        # Exclude timestamp and label columns
        exclude_cols = ['timestamp', 'is_anomaly']
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Select scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()
            
        # Fit and transform
        self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
        
        print(f"Features normalized. Total features: {len(self.feature_cols)}")
        
        return self.df
    
    def save_scaler(self, path='models/scaler.pkl'):
        import os
        os.makedirs('models', exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {path}")
        
    def create_all_features(self, normalize=True):
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        print(f"Starting shape: {self.df.shape}")
        
        self.create_rolling_features(windows=[5, 10, 20])
        self.create_lag_features(lags=[1, 2, 3, 5, 10])
        self.create_difference_features()
        self.create_rate_of_change(periods=[5, 10])
        self.create_statistical_features(window=20)
        self.create_cross_sensor_features()
        self.create_time_based_features()
        
        if normalize:
            self.normalize_features(method='robust')
            self.save_scaler()
            
        print("\n" + "="*60)
        print(f"FEATURE ENGINEERING COMPLETE")
        print(f"Final shape: {self.df.shape}")
        print(f"Total features created: {len(self.feature_cols)}")
        print("="*60)
        
        return self.df, self.feature_cols

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('sensor_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create features
    fe = FeatureEngineering(df)
    df_features, feature_cols = fe.create_all_features()
    
    # Save featured data
    df_features.to_csv('sensor_data_features.csv', index=False)
    print("\nFeatured data saved to 'sensor_data_features.csv'")