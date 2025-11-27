# 📁 Project Structure Overview

## Directory Layout

```
project_datascience/
└── dashboard/                    # Main dashboard application
    ├── api/                      # Backend API (Flask)
    │   ├── app.py                # Flask API server with LSTM model
    │   ├── requirements.txt      # Python dependencies
    │   ├── README.md             # API documentation
    │   └── api_client.js         # Frontend API client
    │
    ├── data/                     # Data files (local only, not in Git)
    │   ├── operational_metrics.csv    # Operational sensor data
    │   ├── seismic_events.csv         # Seismic event records
    │   └── README.md                 # Data documentation
    │
    ├── scripts/                  # Model training scripts
    │   ├── train_lstm_model.py   # Main training script
    │   ├── test_model.py         # Model testing
    │   └── requirements.txt      # Training dependencies
    │
    ├── models/                   # Trained models (local only)
    │   ├── lstm_best_model.h5
    │   └── lstm_simple_model.h5
    │
    ├── docs/                     # Documentation
    │   ├── sprint_review.md      # Sprint documentation
    │   └── Data_dictionary_*.docx # Data dictionaries
    │
    ├── index.html                # Main React dashboard
    ├── lstm_model_ammad.h5       # Production model (in Git)
    ├── README.md                 # Project overview
    ├── SETUP.md                  # Setup instructions
    └── PROJECT_STRUCTURE.md      # This file
```

---

## Key Files

### Frontend
- **`index.html`**: Main React dashboard with:
  - Operational Dashboard (real-time metrics)
  - Risk Dashboard (LSTM predictions)
  - Chart visualizations
  - API integration

### Backend
- **`api/app.py`**: Flask API server providing:
  - `/health` - Health check
  - `/predict/forecast` - 7-day risk forecast
  - `/data/operational` - Operational data
  - `/data/seismic` - Seismic data

### Model
- **`lstm_model_ammad.h5`**: Trained LSTM model for seismic risk prediction
- Input: 24 hours of operational data
- Output: 4-class risk prediction (Green/Yellow/Orange/Red)

---

## Data Flow

```
CSV Files (local) 
    ↓
API Server (app.py)
    ↓ Loads & processes data
    ↓ Uses LSTM model for predictions
    ↓
JSON API Responses
    ↓
Frontend Dashboard (index.html)
    ↓ Fetches via fetch API
    ↓ Displays charts & KPIs
```

---

## What's in Git vs Local

### ✅ In Git (Public)
- `index.html` - Dashboard code
- `api/app.py` - API code
- `lstm_model_ammad.h5` - Model file
- `README.md`, `SETUP.md` - Documentation
- Configuration files

### ❌ NOT in Git (Local Only)
- `data/*.csv` - Sensitive data files
- `models/*.h5` - Other model versions
- Large files (>100MB)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API and model status |
| `/predict/forecast` | POST | Get 7-day risk forecast |
| `/data/operational` | GET | Get operational metrics |
| `/data/seismic` | GET | Get seismic events |
| `/data/latest` | GET | Get latest 24 hours of data |

---

## Technology Stack

- **Frontend:** React (CDN), Chart.js, HTML/CSS
- **Backend:** Flask (Python), TensorFlow/Keras
- **Model:** LSTM Neural Network
- **Deployment:** Render (API), GitHub Pages (Dashboard)

