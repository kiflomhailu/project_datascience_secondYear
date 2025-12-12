# 🚦 Seismic Risk Monitoring Dashboard

**Real-time seismic risk prediction dashboard using CatBoost models**

Inspired by: [project_datascience_secondYear](https://github.com/kiflomhailu/project_datascience_secondYear)

---

## 📋 Overview

Modern, interactive dashboard for monitoring seismic risk in geothermal operations. Features:

- ✅ **3 CatBoost Models**: Event occurrence, magnitude prediction, traffic light classification
- ✅ **React UI**: Beautiful, responsive interface with Chart.js visualizations
- ✅ **Real-time Monitoring**: Track operational metrics and seismic activity
- ✅ **Traffic Light System**: 3-class risk levels (GREEN/YELLOW/RED)
- ✅ **Interactive Charts**: Zoom, pan, and filter data
- ✅ **Flask API**: RESTful backend with model predictions

---

## 🚀 Quick Start

### 1. Start the API Server

```bash
cd dashboard/api
pip install -r requirements.txt
python app.py
```

API will run on: `http://localhost:5000`

### 2. Open the Dashboard

```bash
cd dashboard
# Option 1: Python HTTP server
python -m http.server 8080

# Option 2: VS Code Live Server
# Right-click index.html → "Open with Live Server"
```

Dashboard will open at: `http://localhost:8080`

---

## 📁 Project Structure

```
latest_cop/
├── latest/                                    # Your models and data
│   ├── latest_train_earth.py                # Original Dash dashboard
│   ├── operational_seismic_linear_decay121.csv  # Data
│   ├── seismic_event_occurrence_model_v2.cbm
│   ├── seismic_magnitude_model_v2.cbm
│   ├── seismic_traffic_light_3class_model_v2.cbm
│   ├── train_medians_v2.pkl
│   └── optimal_event_threshold_v2.txt
│
└── dashboard/                                 # New React dashboard
    ├── index.html                            # React frontend
    ├── api/
    │   ├── app.py                           # Flask API
    │   └── requirements.txt                 # Dependencies
    └── README.md                             # This file
```

---

## 🎨 Features

### Operational Dashboard 📊
- Monitor injection flow, pressure, temperature
- Select multiple metrics with checkboxes
- Interactive time-series charts with zoom/pan
- Date range filtering

### Risk Dashboard 🚨
- Real-time event detection
- Magnitude predictions
- Traffic light risk classification (GREEN/YELLOW/RED)
- Event threshold adjustment
- Confusion matrix and performance metrics
- Detailed event table

---

## 📡 API Endpoints

### Health Check
```http
GET /health
```
Returns: API status and model loading state

### Data Info
```http
GET /data/info
```
Returns: Dataset statistics, date range, available variables

### Operational Data
```http
GET /data/operational?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&variables=inj_whp,prod_whp
```
Returns: Operational metrics with filtering

### Events
```http
GET /data/events?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&threshold=0.5
```
Returns: Detected seismic events

### Statistics
```http
GET /statistics?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&threshold=0.5
```
Returns: Event statistics and model metrics

---

## 🛠 Technology Stack

### Frontend
- **React 18** - UI framework (CDN)
- **Chart.js 4.4** - Data visualization
- **Babel Standalone** - JSX transpilation
- **Modern CSS** - Gradient backgrounds, glassmorphism

### Backend
- **Flask 3.0** - Web framework
- **CatBoost** - Machine learning models
- **Pandas** - Data processing
- **NumPy** - Numerical computations
- **scikit-learn** - Metrics and evaluation

---

## 🎯 Models

### 1. Event Occurrence Model
- **Type**: CatBoost Classifier
- **Output**: Binary (event / no event)
- **Threshold**: Configurable (default from optimal_event_threshold_v2.txt)

### 2. Magnitude Model
- **Type**: CatBoost Regressor
- **Output**: Predicted magnitude value
- **Triggered**: Only for predicted events

### 3. Traffic Light Model
- **Type**: CatBoost Classifier (3-class)
- **Output**: 
  - 0 = GREEN (safe, magnitude < 0.17)
  - 1 = YELLOW (warning, 0.17 ≤ magnitude < 1.0)
  - 2 = RED (danger, magnitude ≥ 1.0)

---

## 📊 Data Processing

### Preprocessing Steps:
1. Sentinel value replacement (-999 → 0)
2. Datetime parsing
3. Chronological sorting
4. Feature engineering:
   - Temporal features (hour, day, weekend)
   - Rolling statistics (6h, 12h, 24h windows)
   - Rate of change calculations
   - Pressure/temperature differences
   - Energy efficiency metrics
   - Interaction features

### Feature Count: 50+ engineered features

---

## 🎨 UI Design

### Color Palette
- **Primary Gradient**: #667eea → #764ba2
- **Green (Safe)**: #10b981
- **Yellow (Warning)**: #f59e0b
- **Red (Danger)**: #ef4444

### Design Elements
- Glassmorphism cards with backdrop blur
- Animated gradient background
- Smooth transitions and hover effects
- Responsive grid layouts
- Interactive charts with zoom controls

---

## 📈 Performance

### Data Loading
- Automatic sampling for large datasets (>10k records)
- Efficient date range filtering
- Variable selection to reduce payload

### API Optimization
- CORS enabled for cross-origin requests
- Error handling and validation
- Efficient numpy/pandas operations

---

## 🔧 Configuration

### Change API URL
Edit `index.html` line 383:
```javascript
const API_BASE_URL = 'http://localhost:5000';
```

### Adjust Event Threshold
Use the slider in Risk Dashboard or modify `optimal_event_threshold_v2.txt`

### Add/Remove Metrics
Edit `OPERATIONAL_VARS` in `index.html` (line 371):
```javascript
const OPERATIONAL_VARS = {
  'inj_flow': 'Injection Flow (m³/h)',
  'inj_whp': 'Injection Pressure (bar)',
  // Add more...
};
```

---

## 🐛 Troubleshooting

### API Not Starting
```bash
# Check if models exist
ls ../latest/*.cbm

# Install dependencies
pip install -r requirements.txt

# Run with verbose output
python app.py
```

### Dashboard Not Loading
```bash
# Check API health
curl http://localhost:5000/health

# Try different port
python -m http.server 8081
```

### No Data Displayed
- Check date range (must match data in CSV)
- Verify CSV path in `app.py`
- Check browser console (F12) for errors

---

## 📚 Original Dash Dashboard

Your original Dash dashboard (`latest_train_earth.py`) is still available with:
- Multi-variable selection checkboxes
- Interactive date range picker
- Dynamic threshold adjustment
- Confusion matrix visualization
- Searchable event table (Dash DataTable)
- Actual vs predicted comparison

To run it:
```bash
cd latest
python latest_train_earth.py
# Access at http://127.0.0.1:8050
```

---

## 🎓 Credits

- **Original UI Design**: [kiflomhailu/project_datascience_secondYear](https://github.com/kiflomhailu/project_datascience_secondYear)
- **Models**: CatBoost v2 seismic prediction models
- **Dashboard Framework**: React + Chart.js
- **Backend**: Flask + CatBoost

---

## 📄 License

MIT License - Feel free to use and modify!

---

## 🤝 Contributing

This dashboard combines the best of both worlds:
- **Original repo's UI**: Modern React design
- **Your models**: Advanced CatBoost prediction pipeline

Perfect for seismic monitoring in geothermal operations! 🌋⚡

---

## 📞 Support

If you encounter issues:
1. Check that all models are in `../latest/` folder
2. Verify CSV data path in `api/app.py`
3. Ensure Flask API is running on port 5000
4. Check browser console for JavaScript errors
5. Review API logs for Python errors

**Happy Monitoring! 🚦📊🔥**
