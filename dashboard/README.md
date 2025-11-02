# Geothermal Plant Monitoring Dashboard

Real-time monitoring and risk prediction system for geothermal power plant operations and induced seismicity.

**Built with React 18, Chart.js, and modern web technologies.**

---

## 📁 Project Structure

```
dashboard/
├── index.html                         # Complete React dashboard (single-page app)
│
├── data/                              # Data files
│   ├── seismic_events.csv            # 380 seismic events (2018-2021)
│   ├── operational_metrics.csv       # Time-series injection data (232MB)
│   └── predictions.csv               # ML model outputs (next sprint)
│
├── assets/                            # Static resources
│   └── images/                       # Logos, screenshots, icons
│
├── docs/                              # Documentation
│   ├── Data_dictionary_Operational_metrics.docx
│   ├── Data_dictionary_Seismic_events.docx
│   └── sprint_review.md             # Sprint documentation
│
├── README.md                          # This file
└── STRUCTURE.md                       # Detailed folder structure explanation
```

**Simple & Clean** - Everything you need in one place!

## ✨ Features

### Operational Dashboard
- Real-time monitoring of injection flow and wellhead pressure
- Seismic magnitude overlay on time-series data
- Interactive date range filtering (2018-2025)
- KPI cards: Total events, max magnitude, average metrics

### Risk Dashboard
- 7-day probability forecasts (Yellow/Orange/Red alert levels)
- Visual risk indicator gauge (Low/Medium/High)
- Trend analysis for seismic risk prediction
- Ready for ML model integration

## 🛠️ Tech Stack

- **Framework**: React 18 (CDN-based, no build required)
- **Visualization**: Chart.js 4.4 with dual-axis time-series
- **Styling**: Custom CSS3 with modern design system
- **Data Loading**: JavaScript Fetch API (ready for CSV integration)
- **State Management**: React Hooks (useState, useEffect, useRef)
- **Deployment**: GitHub Pages compatible

## 🚀 Getting Started

### Option 1: Quick Start (Direct Open)
```bash
# Just open the file in your browser
# Double-click: index.html
```

### Option 2: Local Server (Recommended for CSV Loading)
```bash
# Navigate to dashboard folder
cd dashboard

# Start Python server
python -m http.server 8000

# Open in browser
# http://localhost:8000
```

### Option 3: VS Code Live Server
1. Install "Live Server" extension
2. Right-click `index.html`
3. Select "Open with Live Server"

## 📊 Current Status

### ✅ Completed (This Sprint)
- Dashboard UI/UX framework
- Operational & Risk dashboard designs
- Interactive Chart.js visualizations
- Date filtering functionality
- Responsive layout
- Clean folder structure

### 🔄 In Progress
- GitHub Pages deployment
- Sprint review documentation

### 📅 Next Sprint Goals
1. **Load real CSV data** into dashboard
2. **Integrate ML model predictions** from friend's work
3. **Extract insights** from seismic event patterns
4. **Add data export** functionality
5. **Deploy live demo** to GitHub Pages

## 📈 Data Overview

### Seismic Events (`data/seismic_events.csv`)
- **Records**: 380 events (2018-2021)
- **Fields**: Timestamp, magnitude, location (x,y,z), phase, PGV
- **Magnitude Range**: -0.6 to 2.1
- **Key Insight**: Events cluster during active injection phases

### Operational Metrics (`data/operational_metrics.csv`)
- **Size**: 232 MB (time-series data)
- **Fields**: Timestamp, injection flow, wellhead pressure
- **Period**: 2018-2025
- **Note**: Large file, consider sampling for GitHub

### Risk Predictions (`data/predictions.csv`)
- **Status**: Next sprint (ML model integration)
- **Format**: Date, yellow_prob, orange_prob, red_prob
- **Purpose**: 7-day seismic risk forecasting

## 🎓 Academic Project

**Institution**: Hasselt University  
**Program**: Master of Statistics - Data Science  
**Course**: [Course Name/Code]  
**Year**: 2024-2025

### Team
- **Dashboard Development**: [Your Name]
- **ML Modeling**: [Friend's Name]

### Approach
This project follows Agile methodology with bi-weekly sprint reviews, demonstrating:
- Data engineering and visualization skills
- Machine learning for predictive analytics
- Full-stack development capabilities
- Scientific communication and documentation

## 📝 Documentation

- `STRUCTURE.md` - Detailed folder organization
- `docs/sprint_review.md` - Sprint documentation
- `docs/Data_dictionary_*.docx` - Dataset field descriptions

## 🌐 Deployment

**GitHub Pages**: Coming soon  
**Repository**: github.com/kiflomhailu/project_datascience_secondYear

---

*Dashboard framework built for Hasselt University Data Science program - Sprint 1*
