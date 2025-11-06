# Generate synthetic time series sensor data with embedded anomalies

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
class SensorDataGenerator:
    def __init__(self, n_samples=10000, n_sensors=5, anomaly_ratio=0.05, seed=42):
      
        self.n_samples = n_samples
        self.n_sensors = n_sensors
        self.anomaly_ratio = anomaly_ratio
        self.seed = seed
        np.random.seed(seed)
        
    def generate_normal_data(self):
        # Create time index
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(self.n_samples)]
        
        data = {'timestamp': timestamps}
        
        # Generate data for each sensor
        for sensor_id in range(self.n_sensors):
            # Base trend
            trend = np.linspace(20 + sensor_id * 5, 25 + sensor_id * 5, self.n_samples)
            
            # Seasonal component (daily and weekly patterns)
            daily_season = 3 * np.sin(2 * np.pi * np.arange(self.n_samples) / 24)
            weekly_season = 2 * np.sin(2 * np.pi * np.arange(self.n_samples) / (24 * 7))
            
            # Random noise
            noise = np.random.normal(0, 0.5, self.n_samples)
            
            # Combine components
            sensor_values = trend + daily_season + weekly_season + noise
            
            data[f'sensor_{sensor_id}'] = sensor_values
            
        return pd.DataFrame(data)
    
    def inject_anomalies(self, df):
        """Inject different types of anomalies into the data"""
        n_anomalies = int(self.n_samples * self.anomaly_ratio)
        anomaly_indices = np.random.choice(self.n_samples, n_anomalies, replace=False)
        
        # Create anomaly labels
        df['is_anomaly'] = 0
        df.loc[anomaly_indices, 'is_anomaly'] = 1
        
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'drop', 'drift', 'correlation_break'])
            
            if anomaly_type == 'spike':
                # Sudden spike in one or more sensors
                affected_sensors = np.random.choice(sensor_cols, 
                                                   size=np.random.randint(1, len(sensor_cols)+1), 
                                                   replace=False)
                for sensor in affected_sensors:
                    df.loc[idx, sensor] += np.random.uniform(10, 20)
                    
            elif anomaly_type == 'drop':
                # Sudden drop in sensor values
                affected_sensors = np.random.choice(sensor_cols, 
                                                   size=np.random.randint(1, len(sensor_cols)+1), 
                                                   replace=False)
                for sensor in affected_sensors:
                    df.loc[idx, sensor] -= np.random.uniform(10, 20)
                    
            elif anomaly_type == 'drift':
                # Gradual drift over time
                drift_length = min(50, self.n_samples - idx)
                affected_sensor = np.random.choice(sensor_cols)
                drift = np.linspace(0, np.random.uniform(5, 15), drift_length)
                df.loc[idx:idx+drift_length-1, affected_sensor] += drift
                
            elif anomaly_type == 'correlation_break':
                # Break normal correlation between sensors
                if len(sensor_cols) >= 2:
                    sensor1, sensor2 = np.random.choice(sensor_cols, 2, replace=False)
                    df.loc[idx, sensor1] += np.random.uniform(5, 10)
                    df.loc[idx, sensor2] -= np.random.uniform(5, 10)
        
        return df
    
    def generate(self, save_path='sensor_data.csv'):
        """Generate complete dataset with anomalies"""
        print(f"Generating {self.n_samples} samples with {self.n_sensors} sensors...")
        df = self.generate_normal_data()
        df = self.inject_anomalies(df)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
        print(f"Total anomalies injected: {df['is_anomaly'].sum()} ({self.anomaly_ratio*100:.1f}%)")
        
        return df

if __name__ == "__main__":
    # Generate synthetic sensor data
    generator = SensorDataGenerator(n_samples=10000, n_sensors=5, anomaly_ratio=0.05)
    df = generator.generate('sensor_data.csv')
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())