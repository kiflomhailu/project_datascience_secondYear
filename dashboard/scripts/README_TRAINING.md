# LSTM Model Training Guide

## 📁 Project Structure
```
dashboard/
├── data/
│   ├── seismic_events.csv          # 380 seismic events
│   └── operational_metrics.csv     # 232MB operational data
├── scripts/
│   ├── train_simple.py             # Quick start training (RECOMMENDED)
│   ├── train_lstm_model.py         # Full training with 4-class prediction
│   └── requirements.txt            # Python dependencies
└── models/
    └── (trained models will be saved here)
```

## 🚀 Quick Start (5 Steps)

### Step 1: Install Dependencies
```bash
cd scripts
pip install -r requirements.txt
```

### Step 2: Run Simple Training (Fastest)
```bash
python train_simple.py
```
**This will:**
- Load data (50k rows from operational metrics)
- Train LSTM to predict seismic events 24 hours ahead
- Save model to `../models/lstm_simple_model.h5`
- Takes ~5-10 minutes

### Step 3: Run Full Training (Traffic Light System)
```bash
python train_lstm_model.py
```
**This will:**
- Load full datasets
- Train 4-class LSTM (Green/Yellow/Orange/Red)
- Predict risk 7 days ahead
- Save best model to `../models/lstm_best_model.h5`
- Takes ~20-30 minutes

## 📊 What the Model Does

### Input Features (X):
- **Operational**: `inj_flow`, `inj_whp`, `inj_temp`, `prod_temp`, `prod_whp`
- **Seismic**: `event_count`, `max_magnitude`, `avg_magnitude`, `max_pgv`, `avg_pgv`
- **Lookback**: 24 hours of historical data

### Output (y):
**Simple Model:**
- Binary: Will seismic event occur in next 24 hours? (0=No, 1=Yes)

**Full Model (Traffic Light):**
- 4 classes based on magnitude in next 7 days:
  - 🟢 Green (0): magnitude < 0.5
  - 🟡 Yellow (1): 0.5 ≤ magnitude < 1.0
  - 🟠 Orange (2): 1.0 ≤ magnitude < 1.5
  - 🔴 Red (3): magnitude ≥ 1.5

## 🧠 Model Architecture

```
LSTM(128) → Dropout(0.2) →
LSTM(64) → Dropout(0.2) →
LSTM(32) → Dropout(0.2) →
Dense(16) →
Dense(4, softmax)  # 4 risk classes
```

## 📈 Expected Results

**Training Metrics:**
- Training Accuracy: ~75-85%
- Validation Accuracy: ~70-80%
- Test Accuracy: ~70-75%

**Output Files:**
- `lstm_simple_model.h5` - Binary classifier
- `lstm_seismic_risk_model.h5` - 4-class traffic light model
- `scaler.pkl` - Feature scaler for predictions
- `training_history.png` - Training plots

## 🛠️ Troubleshooting

### Memory Error
If you get memory errors with operational_metrics.csv:
1. Open `train_lstm_model.py`
2. Find line: `nrows=100000`
3. Change to: `nrows=50000` or `nrows=20000`

### Missing TensorFlow
```bash
pip install tensorflow==2.13.0
```

### Slow Training
- Use GPU version: `pip install tensorflow-gpu`
- Reduce batch size: Change `batch_size=32` to `batch_size=16`
- Reduce epochs: Change `epochs=50` to `epochs=20`

## 📝 Next Steps After Training

1. **Test predictions:**
```python
from tensorflow.keras.models import load_model
import joblib

model = load_model('../models/lstm_seismic_risk_model.h5')
scaler = joblib.load('../models/scaler.pkl')

# Make predictions on new data
predictions = model.predict(new_data_scaled)
risk_levels = np.argmax(predictions, axis=1)
```

2. **Integrate with dashboard:**
- Load model in React dashboard
- Display real-time risk predictions
- Show 7-day forecast probabilities

3. **Improve model:**
- Add more features (spatial coordinates, time of day)
- Try different architectures (GRU, Transformer)
- Hyperparameter tuning with Grid Search

## 📚 Data Summary

**Seismic Events (seismic_events.csv):**
- 380 events from 2018-2025
- Features: magnitude, PGV, location, timestamp
- Size: ~90KB

**Operational Metrics (operational_metrics.csv):**
- ~2.5M rows of sensor data
- 5-minute intervals
- Features: injection flow/pressure/temp, production metrics
- Size: 232MB

## 🎯 Model Purpose

This LSTM model supports the **Traffic Light System** for:
- **Proactive Risk Management**: Predict seismic events 7 days ahead
- **Operational Adjustments**: Modify injection rates based on risk
- **Safety Compliance**: Alert operators before high-risk periods
- **Data-Driven Decisions**: Replace reactive monitoring with prediction

---

**Team:** Thierry Fotabong, Muhammad Ammad, Laiba Tahir, Tanjim Hossain, Berhe Kiflom, Alain Patrick  
**Course:** Project Data Science - Hasselt University  
**Sprint:** Week 3-4 (Baseline Modeling)
