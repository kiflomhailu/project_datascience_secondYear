# 🔌 API Guide - Complete Documentation

## Overview

The API is a Flask server that connects your LSTM model to the frontend dashboard. It serves predictions, loads data from CSV files, and handles all backend operations.

---

## 🚀 Quick Start

### Start the API
```bash
cd api
python app.py
```

The API will start on `http://localhost:5000`

### Test if it's working
Open in browser: `http://localhost:5000/health`

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

---

## 📡 API Endpoints

### 1. Health Check
**GET** `/health`

Check if API and model are working.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

**Example:**
```bash
curl http://localhost:5000/health
```

---

### 2. Get Operational Data
**GET** `/data/operational`

Get operational metrics (flow, pressure, temperature) for a date range.

**Query Parameters:**
- `start_date` (required): Start date/time in ISO format
  - Example: `2018-11-28T00:00:00`
- `end_date` (required): End date/time in ISO format
  - Example: `2018-12-02T23:59:59`
- `limit` (optional): Max records to return (default: 1000)

**Response:**
```json
{
  "data": [
    {
      "timestamp": "2018-11-28T00:00:00",
      "inj_flow": 12.5,
      "inj_whp": 8.3,
      "inj_temp": 150.0,
      "prod_temp": 120.0,
      "prod_whp": 6.2
    }
  ],
  "count": 100,
  "source": "real_csv"
}
```

**Example:**
```bash
curl "http://localhost:5000/data/operational?start_date=2018-11-28T00:00:00&end_date=2018-12-02T23:59:59"
```

**If no data:**
```json
{
  "data": [],
  "count": 0,
  "message": "No operational data available (CSV files not found)"
}
```

---

### 3. Get Seismic Data
**GET** `/data/seismic`

Get seismic events (earthquakes) for a date range.

**Query Parameters:**
- `start_date` (required): Start date/time
- `end_date` (required): End date/time

**Response:**
```json
{
  "data": [
    {
      "timestamp": "2018-11-28T12:30:00",
      "magnitude": 2.1,
      "pgv_max": 0.05,
      "x": 123.45,
      "y": 67.89,
      "z": -1.2
    }
  ],
  "count": 5,
  "source": "real_csv"
}
```

**Example:**
```bash
curl "http://localhost:5000/data/seismic?start_date=2018-11-28T00:00:00&end_date=2018-12-02T23:59:59"
```

---

### 4. Get Latest Data
**GET** `/data/latest`

Get the latest 24 hours of operational and seismic data (for model predictions).

**Response:**
```json
{
  "operational": [
    {
      "timestamp": "2018-12-02T23:00:00",
      "inj_flow": 10.5,
      "inj_whp": 8.0,
      ...
    }
  ],
  "seismic": [
    {
      "timestamp": "2018-12-02T22:30:00",
      "magnitude": 1.5,
      ...
    }
  ],
  "records_loaded": 24
}
```

**Example:**
```bash
curl http://localhost:5000/data/latest
```

---

### 5. Get Forecast (7-Day Risk Prediction)
**POST** `/predict/forecast`

Get 7-day seismic risk forecast using LSTM model.

**Request Body:**
```json
{
  "current_date": "2018-12-02T12:00:00",
  "historical_data": [
    {
      "timestamp": "2018-12-01T12:00:00",
      "inj_flow": 10.5,
      "inj_whp": 8.0,
      "inj_temp": 150.0,
      "prod_temp": 120.0,
      "prod_whp": 6.2,
      "event_count": 0,
      "max_magnitude": 0.0,
      "avg_magnitude": 0.0,
      "max_pgv": 0.0,
      "avg_pgv": 0.0
    }
    // ... 23 more hours of data
  ]
}
```

**Response:**
```json
{
  "forecast": [
    {
      "date": "2018-12-03",
      "risk_level": "Yellow",
      "risk_level_code": 1,
      "probabilities": {
        "green": 0.1,
        "yellow": 0.6,
        "orange": 0.2,
        "red": 0.1
      }
    }
    // ... 6 more days
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/predict/forecast \
  -H "Content-Type: application/json" \
  -d '{"current_date": "2018-12-02T12:00:00", "historical_data": [...]}'
```

---

## 🔍 How It Works

### 1. Model Loading
When API starts:
- Looks for `lstm_model_ammad.h5` in multiple locations
- Loads the trained LSTM model
- Loads scaler (or creates dummy scaler if not found)

### 2. Data Loading
When you request data:
- Searches for CSV files in multiple paths:
  - `../data/operational_metrics.csv` (from `api/` folder)
  - `../dashboard/data/operational_metrics.csv`
  - `data/operational_metrics.csv`
- Reads CSV and filters by date range
- Returns JSON data

### 3. Predictions
When you request forecast:
- Takes 24 hours of historical data
- Scales features using StandardScaler
- Adjusts features to match model input (47 features)
- Runs LSTM model prediction
- Applies softmax to get probabilities
- Returns 7-day forecast

---

## 📁 File Structure

```
api/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # Basic API docs
└── API_GUIDE.md       # This file (detailed guide)
```

**Model file location:**
- `dashboard/lstm_model_ammad.h5` (one level up from `api/`)

**Data file locations (searched in order):**
1. `../data/operational_metrics.csv`
2. `../dashboard/data/operational_metrics.csv`
3. `data/operational_metrics.csv`

---

## ⚙️ Configuration

### Port
Default: `5000`

For cloud deployment (Render), uses environment variable:
```python
port = int(os.environ.get('PORT', 5000))
```

### CORS
Enabled for all origins (allows frontend to connect):
```python
CORS(app)
```

---

## 🐛 Troubleshooting

### "Model not found"
- Check `lstm_model_ammad.h5` exists in `dashboard/` folder
- Check API logs for file path errors

### "No data found"
- Check CSV files exist in `dashboard/data/` folder
- Check date range matches data in CSV
- Check API logs: `[OK] Loaded X operational records`

### "API not responding"
- Check API is running: `python app.py`
- Check port 5000 is not blocked
- Check firewall settings

### "CORS error" in browser
- CORS is enabled in code
- Make sure API is running
- Check browser console for exact error

---

## 📊 Data Format

### Operational Data Columns
- `recorded_at`: Timestamp
- `inj_flow`: Injection flow rate [m³/h]
- `inj_whp`: Injection wellhead pressure [bar]
- `inj_temp`: Injection temperature [°C]
- `prod_temp`: Production temperature [°C]
- `prod_whp`: Production wellhead pressure [bar]

### Seismic Data Columns
- `occurred_at`: Timestamp
- `magnitude`: Earthquake magnitude
- `pgv_max`: Peak ground velocity
- `x`, `y`, `z`: Coordinates

---

## 🔐 Security Notes

- CSV files are NOT in Git (`.gitignore`)
- Model file IS in Git (for deployment)
- API has no authentication (local development only)
- For production, add authentication

---

## 💡 Tips

1. **Check logs**: API prints helpful messages:
   - `[OK] Model loaded from: ...`
   - `[OK] Loaded X operational records`
   - `[ERROR] Failed to load from ...`

2. **Test endpoints**: Use browser or `curl` to test:
   - Health: `http://localhost:5000/health`
   - Data: `http://localhost:5000/data/operational?start_date=...`

3. **Date format**: Always use ISO format:
   - `2018-11-28T00:00:00` (with T separator)
   - Not: `2018-11-28 00:00:00`

4. **Performance**: 
   - API reads up to 100k rows for date filtering
   - Then limits results to requested amount
   - Large CSV files are handled efficiently

---

## 📚 Related Files

- **Frontend API Client**: `api/api_client.js` - JavaScript class for frontend
- **Main Dashboard**: `index.html` - Uses the API client
- **Model Training**: `scripts/train_lstm_model.py` - How model was created

---

## 🎯 Common Use Cases

### 1. Load Dashboard Data
```javascript
// Frontend code
const data = await api.getOperationalData(
  '2018-11-28T00:00:00',
  '2018-12-02T23:59:59',
  500
);
```

### 2. Get Risk Forecast
```javascript
// Frontend code
const forecast = await api.getForecast(
  new Date().toISOString(),
  historicalData
);
```

### 3. Check API Status
```javascript
// Frontend code
const health = await api.healthCheck();
if (health.model_loaded) {
  // Model is ready
}
```

---

**Need more help?** Check `api/README.md` for basic setup, or see `SETUP.md` in the root folder for full project setup.

