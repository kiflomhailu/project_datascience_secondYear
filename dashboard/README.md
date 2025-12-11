# Geothermal Energy - Seismic Risk Prediction System

**Predictive traffic light system for forecasting seismic risk in geothermal power plant operations.**

## 📋 Overview

This project implements a predictive system to forecast seismic risk levels (Green/Yellow/Red) for geothermal operations. The system integrates **event occurrence**, **magnitude prediction**, and **traffic light classification** into a unified framework, with a real-time dashboard for monitoring and visualization.

## 🚀 Quick Start

### Start API Server
```bash
cd api
pip install -r requirements.txt
python app.py
````

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

* **Frontend:** React, Chart.js
* **Backend:** Flask (Python)
* **Models:** CatBoost (Event Occurrence, Magnitude, 3-Class Traffic Light)
* **Deployment:** Render (API), GitHub Pages (Dashboard)

## 📡 API Endpoints

* `GET /health` - Health check
* `GET /data/operational` - Get operational data
* `GET /data/seismic` - Get seismic events
* `POST /predict/forecast` - Get 7-day risk forecast

## 🎯 Model Details

* **Models Used:**

  1. **Event Occurrence Model:** Predicts probability of a seismic event.
  2. **Magnitude Model:** Estimates magnitude for predicted events.
  3. **3-Class Traffic Light Model:** Classifies risk into **Green, Yellow, Red**.

* **Inputs:**
  Operational features (pressure, temperature, flow rates, cumulative energy, etc.) and temporal features (hour, day, month, weekend flag).

* **Outputs:**

  * Event probability (0–1)
  * Predicted magnitude (for events)
  * Traffic light class (Green, Yellow, Red)

* **Feature Engineering Highlights:**

  * Rolling statistics (mean, std, max)
  * Rate-of-change metrics
  * Pressure and temperature differences
  * Energy efficiency ratios
  * Interaction features (e.g., temperature × pressure)

* **Training Highlights:**

  * COVID-period downsampling for GREEN class (2021–2022)
  * Chronological train-test split
  * CatBoost hyperparameter tuning with early stopping

* **Performance (Test Set):**

  * **Event Occurrence:** AUC 0.999997, F1-score 0.98 for events
  * **Magnitude Prediction:** RMSE 0.26, R² 0.54, correlation 0.97
  * **Traffic Light Classification:** Weighted F1 ~1.0, RED recall 1.0, YELLOW recall 0.91

* **Saved Models:**

  * `seismic_event_occurrence_model_v2.cbm`
  * `seismic_magnitude_model_v2.cbm`
  * `seismic_traffic_light_3class_model_v2.cbm`
  * Training medians for feature imputation (`train_medians_v2.pkl`)
  * Optimal event threshold (`optimal_event_threshold_v2.txt`)

## ✨ Improvements

* 3-class traffic light system (Green, Yellow, Red)
* COVID period GREEN downsampling (2021–2022)
* Fixed indexing for magnitude model
* Integrated predictions for event, magnitude, and traffic light in one CSV
* Ready for operational deployment

## 📋 Example Predictions

The system generates a CSV (`seismic_predictions_v2.csv`) with columns:

* `event_probability` – Probability of a seismic event
* `event_predicted` – Binary event prediction (0/1)
* `magnitude_predicted` – Predicted magnitude (0 for non-events)
* `traffic_light_predicted` – Traffic light class (0=Green,1=Yellow,2=Red)
* `traffic_light_label_pred` – Traffic light label (🟢 GREEN, 🟡 YELLOW, 🔴 RED)
* `event_actual` – Actual event occurrence
* `magnitude_actual` – Actual magnitude
* `traffic_light_actual` – Actual traffic light class
* `traffic_light_label_actual` – Actual traffic light label

