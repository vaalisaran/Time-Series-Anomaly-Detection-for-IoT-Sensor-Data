# NASA Bearing Anomaly Detection

Complete end-to-end machine learning solution for detecting anomalies in NASA bearing dataset using both statistical and deep learning approaches.

## 📁 Project Structure

```
project/
├── data_loader.py                          # Load NASA bearing dataset
├── feature_engineering.py                  # Create time series features
├── statistical_anomaly_detector.py         # Statistical models (Isolation Forest, LOF)
├── deep_learning_anomaly_detector.py       # Deep learning models (Autoencoder, LSTM)
├── visualization.py                        # Visualization functions
├── main.py                                 # Main pipeline
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the NASA bearing dataset from Kaggle:
- **URL**: https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
- Extract to a local directory

### 3. Update Data Path

Edit `main.py` and update the `DATA_PATH` variable:

```python
DATA_PATH = "/path/to/your/nasa_bearing_dataset"
```

Expected structure:
```
nasa_bearing_dataset/
├── Bearing1_1/
│   ├── 2003.10.22.12.06.24
│   ├── 2003.10.22.12.16.24
│   └── ...
├── Bearing1_2/
└── ...
```

### 4. Run Pipeline

```bash
python main.py
```

## 📊 What the Pipeline Does

### Step 1: Data Loading & Exploration
- Loads NASA bearing sensor data
- Handles missing values
- Generates EDA visualizations:
  - Raw sensor readings
  - Correlation matrix
  - Value distributions

### Step 2: Feature Engineering
- Rolling statistics (mean, std, min, max, range)
- Lag features
- Difference features (1st and 2nd order)
- Statistical features (skewness, kurtosis)
- Interaction features between sensors
- Feature scaling (standardization)

### Step 3: Statistical Anomaly Detection

#### Isolation Forest
- **Intuition**: Isolates anomalies by randomly partitioning data. Anomalies require fewer splits to isolate.
- **Parameters**: 100 trees, 5% contamination rate

#### Local Outlier Factor (LOF)
- **Intuition**: Measures local density deviation. Points with substantially lower density than neighbors are anomalies.
- **Parameters**: 20 neighbors, 5% contamination rate

### Step 4: Deep Learning Anomaly Detection

#### Autoencoder
- **Intuition**: Learns to compress and reconstruct normal data. Anomalies have high reconstruction errors.
- **Architecture**: 
  - Encoder: Input → 128 → 64 → 32
  - Decoder: 32 → 64 → 128 → Output
- **Training**: 50 epochs, MSE loss, Adam optimizer

### Step 5: Model Comparison
- Compares all models side-by-side
- Evaluates detection rates and patterns

### Step 6: Report Generation
- Creates comprehensive report with all findings
- Saves all visualizations

## 📈 Output

All results are saved to the `results/` directory:

### Visualizations
1. `1_raw_sensor_data.png` - Raw sensor readings over time
2. `2_correlation_matrix.png` - Sensor correlation heatmap
3. `3_distributions.png` - Sensor value distributions
4. `4_isolation_forest_anomalies.png` - Anomalies detected by Isolation Forest
5. `5_isolation_forest_scores.png` - Anomaly scores timeline
6. `6_pca_visualization.png` - PCA visualization of anomalies
7. `7_autoencoder_reconstruction_error.png` - Reconstruction error plot
8. `8_autoencoder_anomalies.png` - Anomalies detected by Autoencoder
9. `9_model_comparison.png` - Comparison of all models

### Report
- `anomaly_detection_report.txt` - Summary of all findings

## 🔧 Configuration

### Adjust Sample Size

For faster execution (development/testing):
```python
pipeline.run(bearing_name='Bearing1_1', sample_size=5000)
```

For full dataset:
```python
pipeline.run(bearing_name='Bearing1_1', sample_size=None)
```

### Change Bearing

Analyze different bearings:
```python
pipeline.run(bearing_name='Bearing1_2', sample_size=5000)
pipeline.run(bearing_name='Bearing2_1', sample_size=5000)
```

### Adjust Contamination Rate

In `main.py`, modify detector initialization:
```python
if_detector = StatisticalAnomalyDetector(contamination=0.10)  # 10% anomalies
```

### Adjust Deep Learning Parameters

In `main.py`, modify training:
```python
ae_detector.train(
    X, 
    epochs=100,          # More training
    batch_size=128,      # Larger batches
    learning_rate=0.0001 # Lower learning rate
)
```

## 📚 Key Features

### Production-Ready Code
- ✅ Proper error handling
- ✅ Comprehensive logging
- ✅ Modular design
- ✅ Type hints and documentation
- ✅ Clean code structure

### Feature Engineering
- ✅ Time series specific features
- ✅ Multiple window sizes
- ✅ Statistical transformations
- ✅ Automated feature scaling

### Multiple Detection Approaches
- ✅ Unsupervised statistical methods
- ✅ Deep learning approaches
- ✅ Ensemble capability

### Comprehensive Evaluation
- ✅ Multiple metrics
- ✅ Visual validation
- ✅ Domain-driven analysis

## 🎯 Business Insights

### Detected Anomalies Indicate:
1. **Equipment Degradation**: Gradual wear and tear
2. **Potential Failures**: Sudden spikes in readings
3. **Maintenance Needs**: Consistent deviations from normal
4. **Pattern Changes**: Shifts in operational characteristics

### Actionable Recommendations:
- Schedule maintenance when anomaly rates increase
- Investigate specific sensors showing consistent anomalies
- Use temporal patterns to predict failure windows
- Combine multiple model outputs for robust detection

## 🛠️ Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### PyTorch Issues
```bash
# CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU-enabled PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
- Reduce `sample_size` parameter
- Process bearings one at a time
- Use smaller batch sizes for deep learning

### File Not Found Errors
- Verify dataset path is correct
- Check bearing name matches directory name
- Ensure CSV files exist in bearing directory

## 📝 Assignment Requirements Checklist

✅ **Data Preparation & Exploration**
- Load and explore dataset
- Handle missing values
- EDA with visualizations
- Document data quality issues

✅ **Feature Engineering**
- Create meaningful time series features
- Justify feature choices
- Feature normalization/scaling

✅ **Anomaly Detection Models**
- Statistical methods (Isolation Forest, LOF)
- Deep learning method (Autoencoder)
- Explain intuition behind each approach
- Discuss hyperparameter choices

✅ **Model Evaluation**
- Compare approaches with metrics
- Validate results with domain reasoning
- Create visualizations showing detected anomalies

✅ **Deliverables**
- Well-documented code with comments
- Summary document with findings
- 3-4+ EDA and anomaly detection plots
- README explaining how to run code

## 🔍 Further Improvements

### Model Enhancements
- Implement LSTM-Autoencoder for temporal patterns
- Add ensemble voting mechanism
- Implement online learning for streaming data
- Add attention mechanisms for feature importance

### Feature Engineering
- Frequency domain features (FFT, wavelets)
- More sophisticated time series decomposition
- Cross-sensor interaction features
- Domain-specific bearing features

### Evaluation
- Labeled anomaly data for supervised metrics
- Time-to-failure prediction
- Anomaly severity scoring
- Root cause analysis

### Production Deployment
- Real-time inference API
- Model monitoring and drift detection
- Automated retraining pipeline
- Alert system for detected anomalies

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review code comments and documentation
3. Verify all dependencies are installed correctly

## 📄 License

This project is for educational purposes as part of the AI/ML Engineer assignment.