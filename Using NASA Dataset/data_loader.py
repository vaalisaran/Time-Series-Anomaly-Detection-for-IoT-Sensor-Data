import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NASABearingDataLoader:
    def __init__(self, data_path):
        """
        Initialize the NASA Bearing Data Loader.
        Args:
            data_path (str or Path): Root directory containing bearing folders.
        """
        self.data_path = Path(data_path)
        self.data = None

    def load_bearing_data(self, bearing_name='Bearing1_1'):
        """
        Load data for a specific bearing from the dataset folder.
        Each file represents a time step in the bearing test.
        """
        try:
            logger.info(f"📦 Loading {bearing_name} data...")

            bearing_path = self.data_path / bearing_name
            if not bearing_path.exists():
                raise FileNotFoundError(f"Bearing path not found: {bearing_path}")

            # Collect all files (.csv, .txt, or without extension)
            files = sorted(
                list(bearing_path.glob('*')) +
                list(bearing_path.glob('*.txt')) +
                list(bearing_path.glob('*.csv'))
            )
            # Keep only regular files
            files = [f for f in files if f.is_file()]

            if not files:
                raise FileNotFoundError(f"No data files found in {bearing_path}")

            logger.info(f"Found {len(files)} data files in {bearing_path}")

            dataframes = []
            for idx, file_path in enumerate(files):
                try:
                    # Load each file (usually single-column numeric data)
                    df = pd.read_csv(
                        file_path,
                        header=None,
                        sep=None,  # Auto-detect separator (space/comma/tab)
                        engine="python"
                    )

                    # Add metadata columns
                    df['timestamp'] = idx
                    df['file_number'] = idx
                    dataframes.append(df)

                    if (idx + 1) % 100 == 0:
                        logger.info(f"Loaded {idx + 1}/{len(files)} files...")

                except Exception as e:
                    logger.warning(f"⚠️ Skipped file {file_path}: {e}")
                    continue

            # Combine all data
            self.data = pd.concat(dataframes, ignore_index=True)

            # Rename numeric columns to sensor_1, sensor_2, etc.
            sensor_cols = {
                i: f'sensor_{i + 1}'
                for i in range(len(self.data.columns) - 2)
            }
            self.data.rename(columns=sensor_cols, inplace=True)

            logger.info(f"✅ Data loaded successfully. Shape: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")

            return self.data

        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise

    def get_statistics(self):
        """
        Return dataset summary including shape, columns, and missing values.
        """
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
        """
        Handle missing data using different strategies.
        Options: interpolate, forward_fill, drop
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_bearing_data() first.")

        logger.info(f"🛠 Handling missing values using '{strategy}' strategy")

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
        logger.info(f"Missing values: {missing_before} → {missing_after}")

        return self.data


if __name__ == "__main__":
    # Example run
    # 👇 Replace this with your actual dataset path
    data_path = "nasa_bearing_dataset"

    loader = NASABearingDataLoader(data_path)
    data = loader.load_bearing_data('Bearing1_1')

    print("\n=== Dataset Statistics ===")
    print(loader.get_statistics())
