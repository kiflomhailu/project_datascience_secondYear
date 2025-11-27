# 🔌 API - Simple Explanation

## What is the API?

The API is a **bridge** between:
- Your **LSTM model** (Python/TensorFlow)
- Your **Dashboard** (JavaScript/React)

It runs on `http://localhost:5000` and handles all backend work.

---

## 🎯 What Does It Do?

### 1. **Serves Your Model**
- Loads `lstm_model_ammad.h5` when it starts
- Makes predictions when dashboard asks

### 2. **Loads Data from CSV**
- Reads `operational_metrics.csv` and `seismic_events.csv`
- Filters by date range
- Returns JSON to dashboard

### 3. **Makes Predictions**
- Takes 24 hours of data
- Runs through LSTM model
- Returns 7-day risk forecast

---

## 📡 5 Main Endpoints

### 1. `/health` - Is API Working?
```
GET http://localhost:5000/health
```
**Returns:** `{"status": "healthy", "model_loaded": true}`

### 2. `/data/operational` - Get Operational Data
```
GET http://localhost:5000/data/operational?start_date=2018-11-28T00:00:00&end_date=2018-12-02T23:59:59
```
**Returns:** Flow, pressure, temperature data

### 3. `/data/seismic` - Get Seismic Events
```
GET http://localhost:5000/data/seismic?start_date=2018-11-28T00:00:00&end_date=2018-12-02T23:59:59
```
**Returns:** Earthquake events (magnitude, location)

### 4. `/data/latest` - Get Latest 24 Hours
```
GET http://localhost:5000/data/latest
```
**Returns:** Last 24 hours of data (for predictions)

### 5. `/predict/forecast` - Get 7-Day Forecast
```
POST http://localhost:5000/predict/forecast
Body: { "current_date": "...", "historical_data": [...] }
```
**Returns:** 7 days of risk predictions

---

## 🔄 How It Works (Simple Flow)

```
1. Dashboard asks: "Give me data from Nov 28 to Dec 2"
   ↓
2. API reads CSV files
   ↓
3. API filters by date
   ↓
4. API returns JSON
   ↓
5. Dashboard displays in chart
```

**For Predictions:**
```
1. Dashboard asks: "Predict next 7 days"
   ↓
2. API gets last 24 hours of data
   ↓
3. API runs LSTM model
   ↓
4. API returns risk levels (Green/Yellow/Orange/Red)
   ↓
5. Dashboard shows forecast chart
```

---

## 📁 Where Files Are

### API Code
- `api/app.py` - Main API code

### Model File
- `dashboard/lstm_model_ammad.h5` - Your trained model

### Data Files (Local Only)
- `dashboard/data/operational_metrics.csv` - Operational data
- `dashboard/data/seismic_events.csv` - Seismic events

---

## 🚀 How to Use

### Start API
```bash
cd api
python app.py
```

### Test in Browser
Open: `http://localhost:5000/health`

### Test with curl
```bash
curl http://localhost:5000/health
```

---

## ⚠️ Common Issues

### "Model not found"
- Check `lstm_model_ammad.h5` is in `dashboard/` folder

### "No data found"
- Check CSV files are in `dashboard/data/` folder
- Check date range matches data

### "API not responding"
- Make sure API is running (`python app.py`)
- Check port 5000 is not blocked

---

## 💡 Key Points

1. **API = Backend Server** (Python/Flask)
2. **Dashboard = Frontend** (JavaScript/React)
3. **API connects them** (sends data back and forth)
4. **Model runs in API** (not in browser)
5. **Data stays on server** (CSV files not sent to browser)

---

**For detailed docs, see:** `API_GUIDE.md`

