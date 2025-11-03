# Geothermal Energy - Seismic Risk Prediction System

**Predictive traffic light system for forecasting seismic risk in geothermal power plant operations.**

📦 **Repository**: https://github.com/kiflomhailu/project_datascience_secondYear  
🎓 **Institution**: Hasselt University - Master of Statistics (Data Science)  
📅 **Sprint Review**: November 3, 2025 | **Academic Year**: 2024-2025  
🔬 **Status**: Week 3-4 (Baseline Modeling) | **Overall**: ✅ On Track

**Technologies**: Python, Scikit-learn, TensorFlow/Keras (LSTM), Pandas, Matplotlib

---

## 📋 Table of Contents

1. [Research Questions](#-research-questions)
2. [Last Sprint Backlog](#-last-sprint-backlog)
3. [New Insights and Findings](#-new-insights-and-findings)
4. [Proposed Updates to Research Questions](#-proposed-updates-to-research-questions)
5. [Demo and GitHub Status](#-demo-and-github-status)
6. [High Level Planning](#-high-level-planning)
7. [Next Sprint Goals](#-next-sprint-goals)
8. [Risks and Challenges](#-risks-and-challenges)
9. [Team Retrospective](#-team-retrospective)
10. [Tech Stack & Features](#-tech-stack--features)
11. [Getting Started](#-getting-started)

---

## 🎯 Research Questions

### For Previous Sprint:
**Primary Goal**: To develop a **predictive traffic light system** that forecasts seismic risk hours to days ahead, enabling proactive operational adjustments to prevent damaging earthquakes while maximizing geothermal energy production efficiency.

### For This Sprint:
The research question and goal **remain the same** as the previous sprint, as the intended objective was successfully achieved during the earlier phase.

**Focus**: To maintain the same predictive traffic light system for seismic risk forecasting, focusing primarily on:
- Validating model outcomes
- Refining prediction accuracy
- Documenting results and methodology

### Traffic Light System Definition:
- 🟢 **Green**: Low seismic risk - normal operations
- 🟡 **Yellow**: Medium risk (M>1.5) - caution advised
- 🟠 **Orange**: High risk (M>2.5) - reduce injection rates
- 🔴 **Red**: Critical risk (M>3.5) - halt operations

---

## 📝 Last Sprint Backlog

### Status as communicated in previous review:

| Backlog Item | Status | Comments |
|-------------|--------|----------|
| **Explore data sets** | ✅ Done | Successfully analyzed seismic events and operational metrics datasets |
| **Clean the data** | ✅ Done | Handled missing values, null values, and timestamp inconsistencies |
| **Model Selection** | 🔄 In Progress | **LSTM under investigation** - evaluating for time-series prediction |

### Key Achievements:
- ✅ Dataset exploration completed
- ✅ Data cleaning and preprocessing finalized
- 🔄 Model selection ongoing (focus: LSTM for sequential patterns)

---

## 💡 New Insights and Findings

### Data Preprocessing Achievements:
✅ **Successfully completed:**
- **Handled missing values**: Identified and imputed null values in operational metrics
- **Timestamp alignment**: Merged seismic events with operational data on temporal basis
- **Feature engineering**: Created lagged and aggregated features for time-series modeling

### Exploratory Data Analysis (EDA):
✅ **Key discoveries:**
- **Descriptive statistics**: Analyzed distribution of seismic magnitudes and operational parameters
- **Time series trends**: Visualized temporal patterns in injection flow, pressure, and seismic activity
- **Collinearity check**: Identified correlations between operational variables
- **Feature relationships**: Discovered connections between injection operations and seismic responses

### Variable Relationships:
- Strong correlation between injection flow rates and seismic event frequency
- Pressure thresholds identified that correlate with increased seismicity
- Temporal lag observed between operational changes and seismic responses
- Spatial clustering of events near injection well locations

### Model Development Status:
🔄 **Currently investigating:**
- **LSTM (Long Short-Term Memory)** networks for sequential time-series prediction
- Evaluating model architecture for traffic light system classification
- Feature importance analysis for operational parameters

---

## 🔄 Proposed Updates to Research Questions

### Current Status:
The research question and goal **remain unchanged** from the previous sprint. The focus continues to be on developing and refining the predictive traffic light system for seismic risk forecasting.

### Reason for Maintaining Same Objectives:
- Core objective successfully defined and validated
- Model development in progress (LSTM evaluation)
- Focus shifted to implementation and validation rather than redefinition

### Next Phase Focus:
Instead of changing research questions, the team will:
1. **Validate** model predictions against historical data
2. **Refine** LSTM architecture for improved accuracy
3. **Document** methodology and results comprehensively
4. **Optimize** traffic light threshold calibration

---

## 🎬 Demo and GitHub Status

### Current Project Deliverables:

#### � Data Analysis Completed:
- **Dataset Exploration**: Successfully analyzed 380 seismic events (2018-2021) and 232MB operational metrics
- **Data Preprocessing**: Handled missing values, null values, and timestamp alignment
- **Feature Engineering**: Created lagged and aggregated features for time-series modeling
- **EDA Visualizations**: Distribution plots, time series trends, correlation matrices
- **Statistical Analysis**: Descriptive statistics, collinearity checks, variable relationships

#### 🤖 Machine Learning Progress:
- **Baseline Models**: Logistic Regression and Random Forest in development
- **Advanced Models**: LSTM architecture under investigation for sequential pattern recognition
- **Traffic Light Mapping**: Framework designed for risk classification (Green/Yellow/Orange/Red)
- **Model Evaluation**: Preparing confusion matrices, feature importance analysis

#### 📈 Visualization Dashboard:
- **Status**: Framework designed and ready for model integration
- **Features Planned**:
  - Traffic light risk indicator
  - 7-day probability forecasts
  - Confusion matrix displays
  - Feature importance charts
  - Real-time operational monitoring

### GitHub Repository Status:
- ✅ Repository created: `kiflomhailu/project_datascience_secondYear`
- ✅ Data preprocessing scripts committed
- ✅ EDA notebooks and visualizations uploaded
- ✅ Clean folder structure implemented
- ✅ Documentation complete (README, data dictionaries)
- ✅ Data protection configured (.gitignore for CSV files)
- 🔄 Model development code: **In Progress**
- 📅 Dashboard integration: **Planned for next sprint**

**Repository URL**: https://github.com/kiflomhailu/project_datascience_secondYear

### Deployment Status:
- 🔄 **Current Phase**: Model development and validation
- 📅 **Deployment Timeline**: 
  - Week 5-6: Complete LSTM model training
  - Week 7: Deploy dashboard with integrated predictions
  - Target: GitHub Pages for static dashboard hosting
- ⚠️ **Considerations**: Large data files (232MB) require sampling strategy for web deployment

---

## 📊 High Level Planning

### 7-Week Sprint Plan:

| Week | Task Description | Status |
|------|-----------------|---------|
| **Week 1** | **Preprocessing**: Handle missing values & timestamps; merge datasets; create lagged & aggregated features | ✅ **Completed** |
| **Week 2** | **Exploratory Analysis**: Descriptive statistics; visualize trends; check collinearity & correlations | ✅ **Completed** |
| **Week 3-4** | **Baseline Modeling**: Temporal train-test split; train Logistic Regression & Random Forest; map outputs to traffic light system | 🔄 **In Progress** |
| **Week 5-6** | **Advanced Modeling**: Train LSTM/GRU; feature importance & sensitivity analysis; evaluate early-warning capability | 📅 **Planned** |
| **Week 7** | **Reporting & Visualization**: Dashboards (forecasts, confusion matrices, feature importance); final report & recommendations | 📅 **Planned** |

### Overall Status: ✅ **On Track**

#### Progress Summary:
- ✅ **Week 1-2 completed**: Data preprocessing and EDA successfully finished
- 🔄 **Week 3-4 ongoing**: Baseline models under development, LSTM investigation started
- 📅 **Week 5-7 planned**: Advanced modeling and reporting phases scheduled

#### Key Milestones:
- Data quality validated ✅
- Feature engineering completed ✅
- LSTM architecture being evaluated 🔄
- Traffic light classification framework designed 🔄

---

## 📅 Next Sprint Goals

### Next Sprint Backlog:

| Backlog Item | Status | Comments and Action Points |
|-------------|--------|---------------------------|
| **Model Selection** | 🔄 In Progress | Continue LSTM evaluation; finalize architecture for time-series prediction |
| **Dashboard Design** | 🔄 In Progress | Develop traffic light visualization; integrate with model outputs |

### Detailed Action Items:
1. **Complete LSTM Model**:
   - Finalize network architecture (layers, neurons, activation functions)
   - Train on preprocessed dataset
   - Validate prediction accuracy
   - Tune hyperparameters

2. **Dashboard Development**:
   - Design traffic light indicator interface
   - Create real-time forecast visualization
   - Integrate model predictions with UI
   - Add confusion matrix and feature importance displays

3. **Model Validation**:
   - Evaluate early-warning capability (hours to days ahead)
   - Test sensitivity analysis
   - Document performance metrics

4. **Documentation**:
   - Update technical documentation
   - Prepare final report sections
   - Create visualization materials

---

## ⚠️ Risks and Challenges

### Risk Management Table:

| Risk | Impact | Severity | Owner | Status |
|------|--------|----------|-------|--------|
| **Model Selection** | Choosing wrong architecture may delay timeline and reduce accuracy | 🟡 Medium | Team | ⚠️ Unresolved |
| **Frontend Development** | Dashboard delays could impact demonstration and stakeholder feedback | 🟡 Medium | Team | ⚠️ Unresolved |
| **Model Deployment** | Technical issues in production environment may prevent real-time forecasting | 🟡 Medium | Team | ⚠️ Unresolved |
| **Communication** | Poor coordination between team members causes task delays and rework | 🟡 Medium | Team | ⚠️ Unresolved |

### Mitigation Strategies:
- **Model Selection**: Research LSTM best practices; consult with domain experts; run parallel experiments with baseline models
- **Frontend Development**: Allocate dedicated time for UI work; create mockups early; use existing dashboard frameworks
- **Model Deployment**: Test deployment pipeline early; document infrastructure requirements; plan for staging environment
- **Communication**: Establish regular sync meetings; use project management tools; define clear responsibilities

---

## 👥 Team Retrospective

### 😊 What Went Well:
- ✅ **Successfully preprocessed the dataset**: Handled missing values, null values, and timestamp alignment
- ✅ **Conducted initial exploratory analysis**: Visualized distribution and time series trends
- ✅ **Identified variable collinearity**: Analyzed relationships between operational and seismic features
- ✅ **Collaboration was good when tasks were clearly divided**: Team worked effectively on assigned components

### 🤔 Challenges Faced:
- ⚠️ **Time management needs improvement**: Better balance of workload across weeks required
- ⚠️ **Communication between team members**: Delays in tasks due to coordination gaps
- ⚠️ **Model selection complexity**: LSTM architecture requires more investigation time than initially planned

### 💡 Action Items for Next Sprint:
- [ ] Improve time management with realistic task estimations
- [ ] Schedule regular team sync meetings (2x per week recommended)
- [ ] Establish clear communication channels and response times
- [ ] Create shared documentation for model-dashboard integration
- [ ] Define success criteria and deadlines for each task

### 🎯 Team Strengths:
- Successful data preprocessing and cleaning capabilities
- Strong exploratory data analysis skills
- Ability to identify complex variable relationships
- Good collaboration when tasks are well-defined
- Commitment to project goals

---

## 🛠️ Tech Stack & Tools

### Data Science & Machine Learning:
- **Programming Language**: Python 3.x
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn (Logistic Regression, Random Forest)
- **Deep Learning**: TensorFlow/Keras (LSTM, GRU networks)
- **Statistical Analysis**: SciPy, Statsmodels

### Dashboard & Visualization (Planned):
- **Frontend Framework**: React 18 (CDN-based)
- **Visualization Library**: Chart.js 4.4
- **Architecture**: Single-page application
- **Deployment**: GitHub Pages (static hosting)

### Development Tools:
- **Version Control**: Git, GitHub
- **Notebooks**: Jupyter Notebook / Google Colab
- **Environment**: Python virtual environment / Conda
- **Documentation**: Markdown, Word (data dictionaries)

### Data Management:
- **Storage**: Local CSV files (excluded from Git)
- **Size**: 380 seismic events (~90KB), 232MB operational metrics
- **Protection**: .gitignore configuration for sensitive data

---

## 🚀 Getting Started

### For Team Members / Reviewers:

#### Option 1: View Documentation Only
```bash
# Clone the repository
git clone https://github.com/kiflomhailu/project_datascience_secondYear.git
cd project_datascience_secondYear/dashboard

# Read documentation
cat README.md
cat STRUCTURE.md
```

#### Option 2: Run Data Analysis (Requires Data Access)
```bash
# 1. Clone repository
git clone https://github.com/kiflomhailu/project_datascience_secondYear.git
cd project_datascience_secondYear/dashboard

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies (if requirements.txt exists)
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

# 4. Obtain data files from team (not in Git)
# Contact team for: seismic_events.csv, operational_metrics.csv

# 5. Run preprocessing scripts
python scripts/preprocess_data.py

# 6. Run EDA notebooks
jupyter notebook notebooks/exploratory_analysis.ipynb
```

#### Option 3: View Dashboard (When Available)
```bash
# Once deployed, visit:
# https://kiflomhailu.github.io/project_datascience_secondYear/dashboard/
```

### Important Notes:
- **Data Access**: CSV files are excluded from Git (see `.gitignore`). Contact team members for data access.
- **Large Files**: operational_metrics.csv is 232MB and cannot be committed to GitHub.
- **Environment**: Python 3.8+ recommended for compatibility with all libraries.

---

## 📁 Project Structure

```
project_datascience_secondYear/
├── dashboard/
│   ├── data/                              # Data files (excluded from Git)
│   │   ├── seismic_events.csv            # 380 seismic events (2018-2021, 90KB)
│   │   ├── operational_metrics.csv       # Time-series data (232MB)
│   │   └── .gitkeep                      # Keeps folder in Git
│   │
│   ├── scripts/                           # Python preprocessing scripts
│   │   ├── preprocess_data.py            # Data cleaning and feature engineering
│   │   ├── eda_analysis.py               # Exploratory data analysis
│   │   └── model_training.py             # ML model training (in progress)
│   │
│   ├── notebooks/                         # Jupyter notebooks
│   │   ├── exploratory_analysis.ipynb    # EDA visualizations
│   │   ├── baseline_models.ipynb         # Logistic Regression, Random Forest
│   │   └── lstm_development.ipynb        # LSTM model experiments
│   │
│   ├── models/                            # Trained model files
│   │   ├── baseline_lr.pkl               # Logistic Regression model
│   │   ├── baseline_rf.pkl               # Random Forest model
│   │   └── lstm_model.h5                 # LSTM model (when trained)
│   │
│   ├── docs/                              # Documentation
│   │   ├── Data_dictionary_Operational_metrics.docx
│   │   ├── Data_dictionary_Seismic_events.docx
│   │   └── sprint_review.md              # Sprint documentation
│   │
│   ├── assets/                            # Static resources
│   │   └── images/                        # Plots, charts, screenshots
│   │
│   ├── index.html                         # Dashboard interface (planned)
│   ├── README.md                          # This file
│   ├── STRUCTURE.md                       # Detailed folder guide
│   ├── .gitignore                         # Git exclusions (CSV files)
│   └── requirements.txt                   # Python dependencies
│
└── .git/                                  # Git version control
```

**Note**: Data files in `data/` folder are excluded from Git due to size and privacy. Contact team for access.

**Simple & Clean** - Everything you need in one place!


**Simple & Clean** - Everything you need in one place!

---

## ��� Academic Project Information

**Project Title**: Geothermal Energy - Seismic Risk Prediction System  
**Institution**: Hasselt University  
**Program**: Master of Statistics - Data Science  
**Course**: Project Data Science  
**Sprint Review Date**: November 3, 2025  
**Academic Year**: 2024-2025

### Team Members:
- Thierry Fotabong
- Muhammad Ammad
- Laiba Tahir
- Tanjim Hossain
- Berhe Kiflom
- Alain Patrick

### Project Approach:
This project follows **Agile methodology** with bi-weekly sprint reviews, demonstrating:
- ✅ Data engineering and preprocessing skills
- ✅ Exploratory data analysis capabilities
- ✅ Machine learning for predictive analytics (LSTM networks)
- ✅ Dashboard development and visualization
- ✅ Scientific communication and documentation
- ✅ Team collaboration and project management

### Project Goal:
Develop a predictive traffic light system that forecasts seismic risk hours to days ahead, enabling proactive operational adjustments to prevent damaging earthquakes while maximizing geothermal energy production efficiency.

---

## 📚 References and Documentation

- **STRUCTURE.md**: Detailed folder organization guide
- **docs/sprint_review.md**: Complete sprint documentation
- **docs/Data_dictionary_*.docx**: Dataset field descriptions and metadata
- **GitHub Repository**: https://github.com/kiflomhailu/project_datascience_secondYear

---

*Geothermal Energy Seismic Risk Prediction Dashboard - Sprint Review, November 3, 2025*  
*Hasselt University - Master of Statistics (Data Science)*
