# Geothermal Energy - Seismic Risk Prediction System

**Predictive traffic light system for forecasting seismic risk in geothermal power plant operations.**

## 📋 Overview

This project implements an LSTM-based machine learning model to predict seismic risk levels (Green/Yellow/Orange/Red) for geothermal operations, with a real-time dashboard for monitoring and visualization.

## 🚀 Quick Start

### Start API Server
```bash
cd api
pip install -r requirements.txt
python app.py
```

### Start Dashboard
```bash
python -m http.server 8080
```
Open: `http://localhost:8080`

## 📁 Project Structure

```
dashboard/
├── index.html              # Main React dashboard
├── api/
│   ├── app.py             # Flask API server
│   ├── api_client.js      # Frontend API client
│   └── requirements.txt   # Python dependencies
├── scripts/
│   ├── train_lstm_model.py # Model training script
│   └── requirements.txt   # Training dependencies
└── lstm_model_ammad.h5    # Trained LSTM model
```

## 🔧 Technology Stack

- **Frontend:** React, Chart.js
- **Backend:** Flask (Python)
- **Model:** TensorFlow/Keras LSTM
- **Deployment:** Render (API), GitHub Pages (Dashboard)

## 📡 API Endpoints

- `GET /health` - Health check
- `GET /data/operational` - Get operational data
- `GET /data/seismic` - Get seismic events
- `POST /predict/forecast` - Get 7-day risk forecast

## 🎯 Model Details

- **Input:** 24 hours of operational data
- **Output:** 4-class risk prediction (Green/Yellow/Orange/Red)
- **Architecture:** LSTM neural network
