# COMPLETE GUIDE: Merging Data & Building ML Models
## Geothermal Seismic Analysis Project

---

## PART 1: UNDERSTAND YOUR DATA

### You Have Two Datasets:

#### 1. **operational_metrics.csv** (695,625 rows)
- Plant operations recorded every 5 minutes
- Variables: injection flow, pressure, temperature, production metrics
- Time period: November 2018 - September 2025

#### 2. **seismic_events.csv** (378 rows)
- Earthquakes that occurred near the plant
- Variables: magnitude, location, ground velocity
- Same time period

### The Question:
**"Can we predict earthquakes based on plant operations?"**

---

## PART 2: HOW TO MERGE DATA (TIMESTAMP-BASED)

### ✅ RECOMMENDED MERGE STRATEGY

Use **FORWARD MERGE** (Seismic → Operational) for ML modeling

**Why?**
- You want to predict EARTHQUAKES (378 events)
- Each row = one earthquake with its operational context
- Balanced dataset for machine learning

### The Merge Code (Already Done!):

```python
merged = pd.merge_asof(
    left=seismic,                    # Keep earthquakes
    right=ops,                       # Add operational data
    left_on="occurred_at",           # Earthquake time
    right_on="recorded_at",          # Operational time
    direction="nearest",             # Find closest record
    tolerance=pd.Timedelta(minutes=5)
)
```

**Result:** `seismic_operational_improved.csv` (378 rows)

---

## PART 3: MACHINE LEARNING MODELS TO BUILD

### 🎯 THREE ML PROJECTS YOU CAN DO:

---

### **PROJECT 1: EARTHQUAKE MAGNITUDE PREDICTION** (Easiest)

**Question:** Can we predict how strong an earthquake will be based on operations?

**Type:** Regression (predicting a number)

**Target Variable:** `magnitude` (-1.0 to 2.1)

**Features (Inputs):**
- inj_flow (injection flow rate)
- inj_whp (injection wellhead pressure)
- inj_temp (injection temperature)
- prod_flow (production flow)
- prod_temp (production temperature)
- prod_whp (production wellhead pressure)

**Models to Try:**
1. Linear Regression (simplest)
2. Random Forest Regressor (better)
3. XGBoost Regressor (best)

**Difficulty:** ⭐⭐☆☆☆ (Easy)

---

### **PROJECT 2: EARTHQUAKE OCCURRENCE PREDICTION** (Medium)

**Question:** Will an earthquake occur based on current operations?

**Type:** Classification (YES/NO prediction)

**Target Variable:** `has_earthquake` (True/False)

**Dataset:** Use daily_operations_with_earthquake_count.csv
- Each day: Did earthquake occur or not?
- More balanced dataset (some days with, some without)

**Features:**
- Daily average injection parameters
- Daily production metrics
- Trends (increasing/decreasing)

**Models to Try:**
1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. Neural Network

**Difficulty:** ⭐⭐⭐☆☆ (Medium)

---

### **PROJECT 3: TIME SERIES FORECASTING** (Advanced)

**Question:** Can we predict earthquakes hours/days in advance?

**Type:** Time Series Forecasting

**Approach:**
- Use operational data as time series
- Predict future seismic activity
- Models: LSTM, ARIMA, Prophet

**Difficulty:** ⭐⭐⭐⭐⭐ (Hard)

---

## PART 4: MY RECOMMENDATION (STEP-BY-STEP)

### 🌟 START WITH PROJECT 1: Magnitude Prediction

**Why?**
- Easiest to understand
- Clear metrics (R², MAE)
- Good learning project
- Already have the right data!

---

## STEP-BY-STEP IMPLEMENTATION

### STEP 1: Prepare Your Data ✅ (Already Done!)

You have: `seismic_operational_improved.csv`

### STEP 2: Feature Selection

Choose which operational variables to use:
- ✅ inj_flow (injection flow rate)
- ✅ inj_whp (injection wellhead pressure)
- ✅ inj_temp (injection temperature)
- ✅ prod_flow (production flow)
- ✅ prod_temp (production temperature)
- ✅ is_producing (producing or not)

### STEP 3: Train-Test Split

Split data:
- 80% for training (302 earthquakes)
- 20% for testing (76 earthquakes)

### STEP 4: Build Models

Try 3 models and compare:
1. Linear Regression
2. Random Forest
3. XGBoost

### STEP 5: Evaluate

Metrics to use:
- R² Score (higher is better, 0-1)
- Mean Absolute Error (lower is better)
- Root Mean Squared Error (lower is better)

### STEP 6: Feature Importance

Find which operational parameters matter most!

---

## PART 5: ALTERNATIVE APPROACH (IF LIMITED DATA)

### Problem: Only 378 earthquakes might not be enough!

### Solution: CREATE MORE DATA POINTS

**Use Time Windows:**

For each earthquake, get operational data from:
- 1 hour before
- 30 minutes before
- 15 minutes before
- At the moment
- 15 minutes after

This creates 5× more data points! (378 → 1,890)

---

## PART 6: WHICH MERGE TO USE FOR ML?

| ML Project | Use This Dataset | Rows | Why |
|------------|------------------|------|-----|
| **Magnitude Prediction** | seismic_operational_improved.csv | 378 | One per earthquake |
| **Occurrence Prediction** | daily_operations_with_earthquake_count.csv | 2,448 | Balanced YES/NO |
| **Risk Assessment** | operational_during_earthquakes.csv | 681 | High-risk periods |
| **Time Series** | operational_with_earthquakes_FULL.csv | 695,625 | Complete history |

---

## PART 7: HANDLING CHALLENGES

### Challenge 1: Imbalanced Data
**Problem:** More "no earthquake" than "earthquake" data

**Solution:**
- Use SMOTE (synthetic data generation)
- Adjust class weights in model
- Use stratified splitting

### Challenge 2: Missing Values
**Problem:** Some operational data has NaN values

**Solution:**
```python
# Fill missing values
df.fillna(df.mean(), inplace=True)
# Or drop rows with missing values
df.dropna(inplace=True)
```

### Challenge 3: Feature Scaling
**Problem:** Different variables have different ranges

**Solution:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## PART 8: EASY CODE TEMPLATE

### Basic ML Pipeline (Copy-Paste Ready):

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load data
df = pd.read_csv('seismic_operational_improved.csv')

# 2. Select features and target
features = ['inj_flow', 'inj_whp', 'inj_temp', 'prod_flow', 'prod_temp']
X = df[features]
y = df['magnitude']

# 3. Handle missing values
X = X.fillna(X.mean())

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")

# 8. Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(importance)
```

---

## MY FINAL RECOMMENDATION

### 🎯 FOR YOUR PROJECT, DO THIS:

#### **PHASE 1: Data Merging** (Team: Ammad & You)

1. ✅ Use `seismic_operational_improved.csv` (378 rows) - Already done!
2. Clean data: Handle missing values
3. Create additional features:
   - Time of day
   - Day of week
   - Cumulative injection volume
   - Rate of change in pressure

#### **PHASE 2: ML Modeling** (Team: Ammad & You)

**Model 1: Magnitude Prediction**
- Target: `magnitude`
- Features: operational parameters
- Model: Random Forest Regressor
- Goal: Predict earthquake strength

**Model 2: Risk Classification**
- Use daily data
- Target: earthquake occurrence (YES/NO)
- Model: XGBoost Classifier
- Goal: Flag high-risk days

#### **PHASE 3: Validation** (All teams collaborate)

- Cross-validation
- Feature importance analysis
- Compare multiple models
- Document findings

---

## QUICK START (RIGHT NOW):

### Use the 378-row dataset you already have!

**File:** `seismic_operational_improved.csv`

**Next Steps:**
1. Check for missing values
2. Select 5-10 operational features
3. Build Random Forest model
4. Evaluate with R² and MAE
5. Analyze feature importance

**This will take you ~2-3 hours to implement!**

---

## SUMMARY

### ✅ Data Merging: DONE
- Forward merge (seismic → operational)
- 378 earthquakes with operational context
- File: seismic_operational_improved.csv

### ✅ Best ML Approach: Magnitude Prediction
- Predict earthquake magnitude
- Use operational parameters as inputs
- Start with Random Forest
- Easy to understand and implement

### ✅ Timeline:
- Week 1: Data cleaning (Tanjim & Patrick)
- Week 2: Feature engineering & modeling (Ammad & You)
- Week 3: Evaluation & comparison (All)
- Week 4: Documentation & presentation (Thiery & Laiba)

---

## NEED HELP? START HERE:

I can create a complete ML script for you if you want!
Just ask: "Create ML script for magnitude prediction"

---

**Good luck with your project!** 🚀


