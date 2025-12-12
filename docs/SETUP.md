# 🛠️ Setup Guide

Complete installation and setup instructions for the AI-Powered ECG Signal Analysis project.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for model training)
- **Storage**: At least 2GB free space

### Required Software
- Python 3.8+ ([Download](https://www.python.org/downloads/))
- pip (included with Python)
- Git ([Download](https://git-scm.com/downloads))
- Jupyter Notebook or JupyterLab
- Web browser (Chrome, Firefox, or Edge recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/aditanshu/AI-Powered-ECG-Signal-Analysis-for-Cardiac-Biosensors.git
cd AI-Powered-ECG-Signal-Analysis-for-Cardiac-Biosensors
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- **wfdb**: ECG signal processing
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **matplotlib**: Visualization
- **seaborn**: Statistical visualization
- **scikit-learn**: Machine learning utilities
- **tensorflow**: Deep learning framework
- **scipy**: Scientific computing
- **pywavelets**: Wavelet analysis

### 4. Verify Installation

```python
python -c "import wfdb, numpy, pandas, tensorflow; print('All packages installed successfully!')"
```

## Dataset Setup

The MIT-BIH Arrhythmia Database is included in the `data/mitdb/` directory.

### Verify Dataset

```bash
# Check if data directory exists
ls data/mitdb/

# Should show files like: 100.atr, 100.dat, 100.hea, etc.
```

### Download Additional Records (Optional)

```python
import wfdb
# Download specific record
wfdb.dl_database('mitdb', 'data/mitdb', records=['101', '102'])
```

## Running the Project

### Option 1: Jupyter Notebooks

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Navigate to notebooks/**
   - Open `main_analysis.ipynb` for complete pipeline
   - Or run notebooks in sequence:
     1. `01_data_preprocessing.ipynb`
     2. `02_feature_extraction.ipynb`
     3. `03_model_training.ipynb`
     4. `04_visualization.ipynb`

3. **Run cells**
   - Click "Run" or press `Shift + Enter` for each cell
   - Wait for each cell to complete before proceeding

### Option 2: Frontend Applications

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Open HTML files in browser**
   - `index.html` - Main dashboard
   - `realtime_analyzer.html` - Live monitoring
   - `results_viewer.html` - Performance metrics

3. **Use the applications**
   - Upload ECG files (.dat, .csv, .txt)
   - Configure parameters
   - View real-time analysis

## Configuration

### Adjust Model Parameters

Edit in `03_model_training.ipynb`:

```python
# Model hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3

# Data split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
```

### Modify Signal Processing

Edit in `01_data_preprocessing.ipynb`:

```python
# Sampling rate
SAMPLING_RATE = 360  # Hz

# Segment length
SEGMENT_LENGTH = 1.0  # seconds

# Overlap
OVERLAP = 0.5  # 50%
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'wfdb'`

**Solution**:
```bash
pip install wfdb
```

#### 2. TensorFlow GPU Issues

**Problem**: GPU not detected

**Solution**:
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow-gpu

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### 3. Memory Errors

**Problem**: `MemoryError` during model training

**Solution**:
- Reduce batch size: `BATCH_SIZE = 16`
- Use data generators
- Close other applications

#### 4. Jupyter Kernel Issues

**Problem**: Kernel dies during execution

**Solution**:
```bash
# Reinstall ipykernel
pip install --upgrade ipykernel
python -m ipykernel install --user
```

### Dataset Issues

#### Missing Data Files

**Problem**: Cannot find MIT-BIH records

**Solution**:
```python
import wfdb
# Re-download database
wfdb.dl_database('mitdb', 'data/mitdb')
```

#### Corrupted Files

**Problem**: Error reading .dat files

**Solution**:
- Delete corrupted files
- Re-download specific records
- Verify file integrity

## Performance Optimization

### For Faster Training

1. **Use GPU acceleration**
   ```bash
   pip install tensorflow-gpu
   ```

2. **Increase batch size** (if memory allows)
   ```python
   BATCH_SIZE = 64
   ```

3. **Use mixed precision training**
   ```python
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   ```

### For Better Accuracy

1. **Increase epochs**
   ```python
   EPOCHS = 100
   ```

2. **Add data augmentation**
   ```python
   # Add noise, scaling, shifting
   ```

3. **Ensemble models**
   ```python
   # Combine predictions from multiple models
   ```

## Development Setup

### For Contributors

1. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Run tests**
   ```bash
   pytest tests/
   ```

## Next Steps

After successful installation:

1. ✅ Run `01_data_preprocessing.ipynb` to understand data pipeline
2. ✅ Explore `02_feature_extraction.ipynb` for feature engineering
3. ✅ Train models using `03_model_training.ipynb`
4. ✅ Visualize results with `04_visualization.ipynb`
5. ✅ Try frontend applications for interactive analysis

## Getting Help

- **Documentation**: Check [README.md](README.md)
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions

## Additional Resources

- [WFDB Documentation](https://wfdb.readthedocs.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [MIT-BIH Database](https://physionet.org/content/mitdb/)
- [ECG Signal Processing](https://en.wikipedia.org/wiki/Electrocardiography)

---

**Happy Analyzing! 🫀**
