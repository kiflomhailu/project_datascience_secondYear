<<<<<<< HEAD
# Geothermal Plant Seismic Monitoring Project

## 📊 Project Overview

This project analyzes the correlation between geothermal plant operations and seismic activity. We've built interactive dashboards and machine learning models to predict and monitor seismic events based on operational parameters.

## 🏗️ Project Structure

```
project_datascience/
├── dashboard/                 # Interactive web dashboards
│   ├── index.html            # First Sprint Overview Dashboard
│   ├── operational_seismic_dashboard.html  # Operational & Seismic Activity ⭐
│   ├── seismic_risk_dashboard.html         # Risk Assessment & Predictions ⭐
│   └── react_dashboard.html  # React-based dashboard
│
├── python/                   # Data analysis and ML models
│   ├── basic_ml_model.py    # ML model for magnitude prediction
│   ├── reverse_merge.py     # Data merging scripts
│   ├── data_cleaning_comprehensive.py
│   └── *.csv                # Processed datasets
│
├── Data files and dictionary-20250925T180947Z-1-001/  # Original data sources
│   └── Data files and dictionary/
│       ├── seismic_events.csv
│       └── operational_metrics.csv
│
└── Images/                  # Dashboard screenshots and visualizations
```

## 🚀 Key Features

### Dashboards
- **Operational Dashboard**: Real-time monitoring of injection flow, pressure, and seismic events
- **Risk Dashboard**: 7-day probability forecasts for seismic risk levels
- **React Dashboard**: Interactive dashboard with tab navigation

### Machine Learning
- **Magnitude Prediction**: Random Forest model predicting earthquake magnitude from operational parameters
- **Feature Importance Analysis**: Identifies key operational factors affecting seismic activity

### Data Processing
- Merged 378 seismic events with 695,625 operational records
- Comprehensive data cleaning and validation
- Time-series analysis and correlation studies

## 📈 Results

- **Total Seismic Events**: 378
- **Max Magnitude**: 2.1
- **Date Range**: 2018-12-01 to 2025-09-22
- **Operational Records**: 695,625 (5-minute intervals)

## 🛠️ Technologies Used

- **Frontend**: React, Chart.js, HTML5/CSS3
- **Backend**: Python (Pandas, NumPy, Scikit-learn)
- **Visualization**: Matplotlib, Seaborn, Chart.js
- **Data Processing**: Pandas, NumPy

## ⚠️ Important Note

**Sensitive Data Policy:** This repository contains NO sensitive data files. All CSV and data files are excluded via `.gitignore`. Data files remain on local machines only and are never shared or committed to version control.

## 📝 How to Run

### Dashboards
1. Open `dashboard/operational_seismic_dashboard.html` in a web browser
2. Open `dashboard/seismic_risk_dashboard.html` in a web browser
3. Or use `dashboard/react_dashboard.html` for React version

### Python Analysis
```bash
cd python
python basic_ml_model.py
python reverse_merge.py
```

## 📊 Key Findings

- Only 0.1% of operational time had earthquakes nearby
- Injection flow rate and pressure show correlations with seismic magnitude
- ML model achieved good prediction accuracy on test data

## 📅 Project Timeline

- **Sprint 1**: Data integration and dashboard prototyping
- **Future**: Real-time data integration, enhanced ML models

## 📚 Documentation

See `python/README_START_HERE.txt` for detailed project guide.

=======
# project_datascience_secondYear
Project Data Science
>>>>>>> 3d6b2231a81425af9dafe137e48e58e375f24c38
