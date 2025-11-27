# 🚀 Project Setup Guide

## Quick Start (Local Development)

### Prerequisites
- Python 3.8+
- Node.js (optional, for development)
- Git

### Step 1: Install API Dependencies

```bash
cd api
pip install -r requirements.txt
```

### Step 2: Place Model File

Ensure `lstm_model_ammad.h5` is in the `dashboard/` folder (root level).

### Step 3: Place Data Files (Optional - for local testing)

For local development, place CSV files in `dashboard/data/`:
- `operational_metrics.csv`
- `seismic_events.csv`

**Note:** These files are NOT pushed to GitHub (they're in `.gitignore` for security).

### Step 4: Start API Server

```bash
cd api
python app.py
```

You should see:
```
[OK] Model loaded from: ../lstm_model_ammad.h5
Server running on http://0.0.0.0:5000
```

### Step 5: Start Dashboard

Open a new terminal:

```bash
cd dashboard
python -m http.server 8080
```

### Step 6: Open Dashboard

Open in browser: `http://localhost:8080`

---

## Cloud Deployment

### API on Render
- URL: `https://project-datascience-secondyear.onrender.com`
- Status: Deployed and working
- Model: Loaded successfully
- Data: Returns empty arrays (CSV files not included for security)

### Dashboard on GitHub Pages
- URL: `https://kiflomhailu.github.io/project_datascience_secondYear/dashboard/`
- Status: Deployed
- Connected to: Render API

---

## Configuration

### API URL Configuration

Edit `index.html` line 250:

**For Local Development:**
```javascript
const API_BASE_URL = 'http://localhost:5000';
```

**For Cloud Deployment:**
```javascript
const API_BASE_URL = 'https://project-datascience-secondyear.onrender.com';
```

---

## Troubleshooting

### API Not Connecting
1. Check if API is running: `http://localhost:5000/health`
2. Check browser console (F12) for CORS errors
3. Ensure API and dashboard are on correct ports

### No Data Showing
- **Local:** Ensure CSV files are in `dashboard/data/` folder
- **Cloud:** Data files are not included (by design for security)
- Dashboard will show "No data available" but model predictions still work

### Model Not Loading
- Check if `lstm_model_ammad.h5` exists in `dashboard/` folder
- Check API logs for model loading errors

---

## Project Structure

```
dashboard/
├── api/                    # Flask API server
│   ├── app.py             # Main API application
│   ├── requirements.txt   # Python dependencies
│   └── README.md          # API documentation
├── data/                   # CSV data files (local only, not in Git)
│   ├── operational_metrics.csv
│   └── seismic_events.csv
├── index.html             # Main dashboard (React)
├── lstm_model_ammad.h5    # Trained LSTM model
└── README.md              # Project overview
```

---

## Next Steps

1. **For Presentation:** Use local setup with data files
2. **For Demo:** Use cloud deployment (works without data files)
3. **For Development:** Use local setup for faster iteration

