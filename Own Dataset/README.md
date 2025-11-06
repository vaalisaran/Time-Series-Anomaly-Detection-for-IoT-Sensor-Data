# Time Series Anomaly Detection for IoT Sensors

A complete end-to-end machine learning solution for detecting anomalies in time series sensor data from IoT devices in manufacturing facilities.

## 📁 Project Structure

```
project/
├── data_generator.py              # Generate synthetic sensor data
├── data_preparation.py            # Data loading, cleaning, and EDA
├── feature_engineering.py         # Feature creation and normalization
├── statistical_anomaly_detector.py # Statistical methods (IF, LOF, etc.)
├── deep_learning_anomaly_detector.py # Neural network methods
├── model_evaluation.py            # Evaluation and comparison
├── main.py                        # Main execution pipeline
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── plots/                         # Generated visualizations
├── models/                        # Saved models
└── results/                       # Final results
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Run everything with one command
python main.py
```

This will:
- Generate synthetic sensor data with embedded anomalies
- Perform exploratory data analysis
- Create meaningful features
- Train 6 different anomaly detection models
- Evaluate and compare all models
- Generate visualizations and reports

### 3. Run Individual Components

```bash
# Generate data only
python data_generator.py

# Run EDA only
python data_preparation.py

# Create features only
python feature_engineering.py

# Train statistical models only
python statistical_anomaly_detector.py

# Train deep learning models only
python deep_learning_anomaly_detector.py

# Evaluate models only
python model_evaluation.py
```

## 📊 Anomaly Detection Methods

### Statistical/Unsupervised Methods

1. **Isolation Forest**
   - Uses ensemble of isolation trees
   - Anomalies are easier to isolate from the rest
   - Fast and effective for high-dimensional data

2. **Local Outlier Factor (LOF)**
   - Density-based method
   - Compares local density of point with neighbors
   - Good for detecting local anomalies

3. **Elliptic Envelope**
   - Assumes Gaussian distribution
   - Uses Mahalanobis distance
   - Best for data following normal distribution

4. **Z-Score Method**
   - Simple statistical approach
   - Identifies points beyond threshold std deviations
   - Fast baseline method

### Deep Learning Methods

1. **Autoencoder**
   - Neural network that learns to reconstruct data
   - High reconstruction error indicates anomalies
   - Captures complex non-linear patterns

2. **LSTM Autoencoder**
   - Captures temporal dependencies
   - Learns sequential patterns
   - Best for time series with temporal correlations

## 📈 Features Created

The pipeline creates over 100+ features including:

- **Rolling Statistics**: Mean, std, min, max (windows: 5, 10, 20)
- **Lag Features**: Previous values (lags: 1, 2, 3, 5, 10)
- **Differences**: First and second order differences
- **Rate of Change**: Percentage changes over time
- **Statistical Features**: Skewness, kurtosis
- **Cross-Sensor Features**: Ratios, differences between sensors
- **Time-Based Features**: Hour, day of week, cyclical encoding

## 📋 Outputs

### Data Files
- `sensor_data.csv` - Raw sensor data with embedded anomalies
- `sensor_data_features.csv` - Data with engineered features
- `results/final_results.csv` - Complete results with all predictions

### Models
- `models/statistical_models.pkl` - Statistical models
- `models/autoencoder.h5` - Trained autoencoder
- `models/lstm_autoencoder.h5` - Trained LSTM autoencoder
- `models/scaler.pkl` - Feature scaler

### Visualizations
- `plots/time_series.png` - Time series for all sensors
- `plots/distributions.png` - Value distributions
- `plots/correlation_matrix.png` - Sensor correlations
- `plots/boxplots.png` - Outlier visualization
- `plots/confusion_matrices.png` - Model performance
- `plots/roc_curves.png` - ROC curves comparison
- `plots/pr_curves.png` - Precision-Recall curves
- `plots/anomaly_timeline.png` - Detected anomalies in context

### Reports
- `evaluation_report.txt` - Detailed model comparison
- `results/model_comparison.csv` - Metrics for all models
- `pipeline.log` - Execution logs

## 🎯 Model Evaluation Metrics

Each model is evaluated using:
- **Precision**: Accuracy of anomaly predictions
- **Recall**: Ability to find all anomalies
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/False positives and negatives
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve

## 🔧 Configuration

### Modify Data Generation

Edit parameters in `main.py`:
```python
run_pipeline(
    n_samples=10000,    # Number of time points
    n_sensors=5,        # Number of sensors
    anomaly_ratio=0.05  # 5% anomalies
)
```

### Hyperparameters

**Isolation Forest:**
- `n_estimators`: 100 (number of trees)
- `contamination`: 0.05 (expected anomaly ratio)

**LOF:**
- `n_neighbors`: 20 (number of neighbors to consider)

**Autoencoder:**
- `epochs`: 50
- `batch_size`: 128
- `encoding_dim`: 32
- `hidden_layers`: [128, 64, 32]

**LSTM Autoencoder:**
- `epochs`: 30
- `timesteps`: 10
- `encoding_dim`: 32

## 📝 Code Quality Features

- **Error Handling**: Try-catch blocks for robustness
- **Logging**: Comprehensive logging throughout pipeline
- **Modularity**: Each component is independent and reusable
- **Documentation**: Detailed docstrings and comments
- **Type Hints**: Clear function signatures
- **Reproducibility**: Random seeds for consistent results

## 🎓 Key Insights

1. **Feature Engineering is Critical**: Created features significantly improve detection
2. **Ensemble Methods Work**: Combining multiple models reduces false positives
3. **Deep Learning Excels**: Autoencoders capture complex patterns
4. **Context Matters**: LSTM helps when temporal patterns are important
5. **Trade-offs Exist**: Balance precision vs recall based on use case

## 🔍 Business Recommendations

1. **Production Deployment**: Use ensemble of Isolation Forest + Autoencoder
2. **Real-time Monitoring**: Isolation Forest for speed
3. **High Accuracy Needs**: LSTM Autoencoder for complex patterns
4. **Cost-sensitive**: Start with Z-score method, upgrade if needed

## 🚧 Limitations & Future Work

**Current Limitations:**
- Synthetic data (test with real sensor data)
- Fixed window sizes (could be adaptive)
- Binary classification (could add severity levels)
- No concept drift handling

**Future Improvements:**
- Online learning for real-time adaptation
- Ensemble stacking with meta-learner
- Attention mechanisms for interpretability
- Anomaly type classification
- Integration with alerting systems
- A/B testing framework

## 📞 Support

For questions or issues:
1. Check the `pipeline.log` file
2. Review `evaluation_report.txt`
3. Examine plots in `plots/` directory

## 📄 License

This project is for educational and evaluation purposes.

---

**Total Execution Time:** ~5-10 minutes (depending on hardware)  
**Python Version:** 3.7+  
**TensorFlow Version:** 2.8+