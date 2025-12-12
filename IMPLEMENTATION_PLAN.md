# Dashboard Implementation Plan
## Adapting GitHub Repository UI to Your Local Models

---

## 📋 Project Overview

**Goal**: Create a React-based dashboard with the GitHub repository's UI structure, but using your local CatBoost models and data.

**Source UI**: https://github.com/kiflomhailu/project_datascience_secondYear.git (main branch)

**Your Models**: 
- `seismic_event_occurrence_model_v2.cbm`
- `seismic_magnitude_model_v2.cbm`
- `seismic_traffic_light_3class_model_v2.cbm`

---

## 📁 Proposed New Structure

```
latest_cop/
├── latest/                                    # Your existing models and data
│   ├── latest_train_earth.py                # Original Dash dashboard
│   ├── operational_seismic_linear_decay121.csv
│   ├── seismic_event_occurrence_model_v2.cbm
│   ├── seismic_magnitude_model_v2.cbm
│   ├── seismic_traffic_light_3class_model_v2.cbm
│   ├── train_medians_v2.pkl
│   └── optimal_event_threshold_v2.txt
│
└── dashboard/                                 # New React dashboard (GitHub UI style)
    ├── index.html                            # React frontend (adapted)
    ├── api/
    │   ├── app.py                           # Flask API (adapted for your models)
    │   ├── api_client.js                    # Frontend API client (from GitHub)
    │   └── requirements.txt                 # Python dependencies
    └── README.md                             # Dashboard documentation
```

---

## 🔄 Implementation Steps

### Phase 1: Setup (15 minutes)
- [x] ✅ Clone GitHub repository structure analysis
- [x] ✅ Understand your current models and data
- [ ] Create `dashboard` folder structure
- [ ] Copy HTML template from GitHub repo
- [ ] Set up Flask API structure

### Phase 2: Backend Adaptation (30 minutes)
- [ ] Create `api/app.py` with Flask setup
- [ ] Implement data loading from your CSV
- [ ] Add preprocessing pipeline (sentinel values, datetime, feature engineering)
- [ ] Load your 3 CatBoost models
- [ ] Create API endpoints:
  - `GET /health` - Health check
  - `GET /data/operational` - Get operational metrics
  - `GET /data/seismic` - Get seismic predictions
  - `POST /predict/forecast` - Generate forecast
- [ ] Add CORS support for frontend

### Phase 3: Frontend Adaptation (30 minutes)
- [ ] Copy `index.html` from GitHub repo
- [ ] Update API endpoint URLs
- [ ] Adapt feature names to match your data:
  - `inj_flow`, `inj_whp`, `inj_temp`, etc.
  - `pgv_max`, `magnitude`, `hourly_seismicity_rate`
- [ ] Change risk levels from 4-class to 3-class:
  - GREEN (0) = Safe
  - YELLOW (1) = Warning
  - RED (2) = Danger
- [ ] Update chart labels and colors
- [ ] Adjust metric checkboxes

### Phase 4: Testing (20 minutes)
- [ ] Start Flask API server
- [ ] Open dashboard in browser
- [ ] Test operational metrics display
- [ ] Test seismic predictions
- [ ] Test traffic light system
- [ ] Verify all charts render correctly

### Phase 5: Polish (15 minutes)
- [ ] Add error handling
- [ ] Improve loading states
- [ ] Add data refresh capability
- [ ] Create README documentation
- [ ] Test on different browsers

**Total Estimated Time: ~2 hours**

---

## 🎨 UI Components to Adapt

### 1. Operational Dashboard Tab
**From GitHub**: 
- Flow rate, pressure, temperature charts

**Your Adaptation**:
```javascript
const operationalMetrics = [
  { key: 'inj_flow', label: 'Injection Flow (m³/h)', color: '#3B82F6' },
  { key: 'inj_whp', label: 'Injection Pressure (bar)', color: '#10B981' },
  { key: 'inj_temp', label: 'Injection Temperature (°C)', color: '#F59E0B' },
  { key: 'prod_temp', label: 'Production Temperature (°C)', color: '#EF4444' },
  { key: 'prod_whp', label: 'Production Pressure (bar)', color: '#8B5CF6' },
  // Add more from your CSV columns
];
```

### 2. Risk Dashboard Tab
**From GitHub**:
- Event count, magnitude, PGV charts
- 4-class forecast (Green/Yellow/Orange/Red)

**Your Adaptation**:
```javascript
const seismicMetrics = [
  { key: 'event_probability', label: 'Event Probability', color: '#EF4444' },
  { key: 'magnitude_predicted', label: 'Predicted Magnitude', color: '#F59E0B' },
  { key: 'pgv_max', label: 'Peak Ground Velocity', color: '#8B5CF6' },
];

const riskLevels = [
  { value: 0, label: 'GREEN', color: '#10B981', description: 'Safe' },
  { value: 1, label: 'YELLOW', color: '#F59E0B', description: 'Warning' },
  { value: 2, label: 'RED', color: '#EF4444', description: 'Danger' }
];
```

---

## 🔧 Key Code Adaptations

### Flask API Structure (`api/app.py`)

```python
from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import pickle

app = Flask(__name__)
CORS(app)

# Load models
model_event = CatBoostClassifier()
model_event.load_model('../latest/seismic_event_occurrence_model_v2.cbm')

model_magnitude = CatBoostRegressor()
model_magnitude.load_model('../latest/seismic_magnitude_model_v2.cbm')

model_traffic = CatBoostClassifier()
model_traffic.load_model('../latest/seismic_traffic_light_3class_model_v2.cbm')

# Load data
df = pd.read_csv('../latest/operational_seismic_linear_decay121.csv')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': 3})

@app.route('/data/operational')
def get_operational_data():
    # Preprocessing and feature engineering
    # ... (from your latest_train_earth.py)
    return jsonify(data)

@app.route('/predict/forecast', methods=['POST'])
def predict_forecast():
    # Make predictions using your 3 models
    # Return forecast data
    return jsonify(forecast)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Frontend API Client (`index.html`)

```javascript
const API_BASE_URL = 'http://localhost:5000';

async function fetchOperationalData() {
  const response = await fetch(`${API_BASE_URL}/data/operational`);
  return await response.json();
}

async function fetchForecast() {
  const response = await fetch(`${API_BASE_URL}/predict/forecast`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  return await response.json();
}
```

---

## 📊 Data Mapping

### Your CSV Columns → Dashboard Metrics

| Your Column | Dashboard Display | Chart Type |
|-------------|-------------------|------------|
| `inj_flow` | Injection Flow | Line chart |
| `inj_whp` | Injection Pressure | Line chart |
| `inj_temp` | Injection Temperature | Line chart |
| `prod_temp` | Production Temperature | Line chart |
| `prod_whp` | Production Pressure | Line chart |
| `event_probability` | Event Probability | Area chart |
| `magnitude_predicted` | Predicted Magnitude | Scatter plot |
| `traffic_light_pred` | Risk Level | Bar chart |

---

## 🎯 Expected Features

### ✅ What You'll Get:

1. **Modern React UI**: Clean, responsive design
2. **Interactive Charts**: Zoom, pan, tooltip interactions
3. **Two Dashboard Tabs**: Operational and Risk views
4. **Real-time Updates**: Refresh button for new data
5. **Metric Selection**: Toggle metrics on/off
6. **3-Class Traffic Light**: Green/Yellow/Red risk levels
7. **Forecast Display**: Prediction timeline
8. **Model Integration**: Your 3 CatBoost models working together
9. **Professional Look**: Same style as GitHub repository
10. **Single HTML File**: Easy deployment, no build process

### 🎨 Visual Design:

- **Color Scheme**: Modern blue/green/orange/red palette
- **Layout**: CSS Grid for responsive design
- **Typography**: Clean sans-serif fonts
- **Charts**: Professional Chart.js visualizations
- **Spacing**: Generous padding and margins
- **Shadows**: Subtle box shadows for depth

---

## 🚀 How to Run

### Start Backend:
```bash
cd dashboard/api
pip install -r requirements.txt
python app.py
# API runs on http://localhost:5000
```

### Start Frontend:
```bash
cd dashboard
python -m http.server 8080
# Open http://localhost:8080 in browser
```

### Alternative (VS Code Live Server):
```bash
# Right-click index.html → "Open with Live Server"
```

---

## 📦 Dependencies

### Backend (`api/requirements.txt`):
```
flask>=3.0.0
flask-cors>=4.0.0
numpy>=1.24.0
pandas>=2.0.0
catboost>=1.2.0
scikit-learn>=1.3.0
```

### Frontend (CDN - no install needed):
- React 18
- Chart.js 4.4
- All loaded from CDN links in HTML

---

## 🔍 Quality Checks

Before finalizing, verify:
- [ ] All 3 models load correctly
- [ ] CSV data loads without errors
- [ ] Feature engineering matches training
- [ ] Predictions are reasonable
- [ ] Charts display properly
- [ ] No console errors
- [ ] Responsive on mobile
- [ ] Colors match risk levels
- [ ] Loading states work
- [ ] Error handling present

---

## 💡 Next Steps

**Ready to build?** I can:

1. ✅ Create the complete dashboard structure
2. ✅ Adapt the Flask API for your models
3. ✅ Modify the HTML frontend
4. ✅ Set up all necessary files
5. ✅ Test the integration

**Would you like me to start building now?** 🚀

Just say "yes" or "start building" and I'll create the complete dashboard for you!
