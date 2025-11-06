

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NASABearingDataLoader:
    
    def __init__(self, data_path):
        ta_path = Path(data_path)
        self.data = None
        
    def load_bearing_data(self, bearing_name='Bearing1_1'):
        
        try:
            logger.info(f"Loading {bearing_name} data...")
            
            bearing_path = self.data_path / bearing_name
            
            if not bearing_path.exists():
                raise FileNotFoundError(f"Bearing path not found: {bearing_path}")
            
            # Get all CSV files
            csv_files = sorted(bearing_path.glob('*.csv'))
            
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {bearing_path}")
            
            logger.info(f"Found {len(csv_files)} files")
            
            # Load all files
            dataframes = []
            for idx, file_path in enumerate(csv_files):
                try:
                    df = pd.read_csv(file_path, header=None)
                    
                    # Add timestamp (assuming each file is a time step)
                    df['timestamp'] = idx
                    df['file_number'] = idx
                    
                    dataframes.append(df)
                    
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Loaded {idx + 1}/{len(csv_files)} files")
                        
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    continue
            
            # Concatenate all dataframes
            self.data = pd.concat(dataframes, ignore_index=True)
            
            # Rename columns
            num_sensors = len([col for col in self.data.columns if col not in ['timestamp', 'file_number']])
            sensor_cols = {i: f'sensor_{i+1}' for i in range(num_sensors)}
            self.data.rename(columns=sensor_cols, inplace=True)
            
            logger.info(f"Data loaded successfully: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_statistics(self):
        if self.data is None:
            raise ValueError("No data loaded. Call load_bearing_data() first.")
        
        stats = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'basic_stats': self.data.describe().to_dict()
        }
        
        return stats
    
    def handle_missing_values(self, strategy='interpolate'):
        
        if self.data is None:
            raise ValueError("No data loaded.")
        
        logger.info(f"Handling missing values using {strategy} strategy")
        
        missing_before = self.data.isnull().sum().sum()
        
        if strategy == 'interpolate':
            sensor_cols = [col for col in self.data.columns if 'sensor' in col]
            self.data[sensor_cols] = self.data[sensor_cols].interpolate(method='linear')
        elif strategy == 'forward_fill':
            self.data.fillna(method='ffill', inplace=True)
            self.data.fillna(method='bfill', inplace=True)
        elif strategy == 'drop':
            self.data.dropna(inplace=True)
        
        missing_after = self.data.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        return self.data


if __name__ == "__main__":
    # Example usage
    data_path = "path/to/nasa/bearing/dataset"
    
    loader = NASABearingDataLoader(data_path)
    data = loader.load_bearing_data('Bearing1_1')
    
    print(loader.get_statistics())