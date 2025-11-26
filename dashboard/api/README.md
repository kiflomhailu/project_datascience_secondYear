# LSTM Model API - Setup Guide

This API connects your trained LSTM model (`lstm_model_ammad.h5`) to the frontend dashboard.

## 📋 Prerequisites

- Python 3.8+
- Trained LSTM model file (`lstm_model_ammad.h5`)
- Scaler file (`scaler.pkl`) - optional but recommended

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd api
pip install -r requirements.txt
```

### 2. Place Model Files

Make sure your model file is accessible. The API will look for:
- `lstm_model_ammad.h5` (in dashboard folder or current directory)
- `scaler.pkl` (in models folder or current directory)

You can also update the paths in `app.py` if your files are elsewhere.

### 3. Run the API Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 4. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Test prediction (requires 24 hours of data)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [/* 24 hours of data */]}'
```

## 📡 API Endpoints

### `GET /health`
Check if API and model are loaded correctly.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### `POST /predict`
Get a single risk prediction from 24 hours of data.

**Request Body:**
```json
{
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00",
      "inj_flow": 0.5,
      "inj_whp": 0.8,
      "inj_temp": 150.0,
      "prod_temp": 120.0,
      "prod_whp": 0.6,
      "event_count": 0,
      "max_magnitude": 0.0,
      "avg_magnitude": 0.0,
      "max_pgv": 0.0,
      "avg_pgv": 0.0
    }
    // ... 23 more hours
  ]
}
```

**Response:**
```json
{
  "risk_level": "Yellow",
  "risk_level_code": 1,
  "probabilities": {
    "green": 0.1,
    "yellow": 0.6,
    "orange": 0.2,
    "red": 0.1
  },
  "confidence": 0.6
}
```

### `POST /predict/batch`
Get predictions for multiple time periods.

### `POST /predict/forecast`
Get 7-day forecast predictions.

## 🔌 Frontend Integration

### Option 1: Use the API Client (Recommended)

Include `api_client.js` in your HTML:

```html
<script src="api/api_client.js"></script>
<script>
  const api = new SeismicRiskAPI('http://localhost:5000');
  
  // Get prediction
  const data = /* your 24 hours of data */;
  const prediction = await api.predict(data);
  console.log(prediction.risk_level);
</script>
```

### Option 2: Direct Fetch

```javascript
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ data: yourData })
});
const result = await response.json();
```

## 🔧 Configuration

### Change API Port

Edit `app.py`:
```python
app.run(host='0.0.0.0', port=5000, debug=True)  # Change port here
```

### Update Model Paths

Edit the `load_model_and_scaler()` function in `app.py` to point to your model files.

### CORS Settings

If deploying to a different domain, update CORS in `app.py`:
```python
CORS(app, origins=['https://your-frontend-domain.com'])
```

## 🐳 Production Deployment

### Using Gunicorn (Recommended)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 📝 Notes

- The model expects exactly 24 hours of historical data
- All features must be provided (missing values default to 0)
- Risk levels: Green (0), Yellow (1), Orange (2), Red (3)
- If scaler is not found, the API will use default scaling (may affect accuracy)

## 🐛 Troubleshooting

**Model not found:**
- Check file paths in `app.py`
- Ensure model file exists and is readable

**Import errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

**CORS errors:**
- Update CORS settings in `app.py`
- Check browser console for specific errors

**Prediction errors:**
- Verify data format matches expected structure
- Ensure you have exactly 24 hours of data
- Check that all required features are present

## 📞 Support

For issues or questions, check:
1. Model training script for expected input format
2. API logs for error messages
3. Browser console for frontend errors

