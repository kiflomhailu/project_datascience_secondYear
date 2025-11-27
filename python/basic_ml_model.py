"""
BASIC MACHINE LEARNING MODEL
Predict earthquake magnitude from operational parameters

Usage: python basic_ml_model.py

This script will:
1. Load merged seismic-operational data
2. Train a Random Forest model
3. Make predictions
4. Show accuracy metrics
5. Display feature importance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

print("="*70)
print("EARTHQUAKE MAGNITUDE PREDICTION MODEL")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\nSTEP 1: Loading data...")

try:
    df = pd.read_csv('seismic_operational_improved.csv')
    print(f"  Loaded: {len(df)} earthquakes")
    print(f"  Columns: {len(df.columns)}")
except FileNotFoundError:
    print("  ERROR: seismic_operational_improved.csv not found!")
    print("  Make sure the file is in the current directory.")
    exit()

# ============================================================================
# STEP 2: SELECT FEATURES
# ============================================================================
print("\nSTEP 2: Selecting features...")

# Define features (operational parameters)
feature_columns = [
    'inj_flow',      # Injection flow rate
    'inj_whp',       # Injection wellhead pressure
    'inj_temp',      # Injection temperature
    'inj_ap',        # Annular pressure
    'prod_flow',     # Production flow
    'prod_temp',     # Production temperature
    'prod_whp',      # Production wellhead pressure
]

# Target variable (what we want to predict)
target = 'magnitude'

# Check which features are available
available_features = [f for f in feature_columns if f in df.columns]
print(f"  Available features: {len(available_features)}")
print(f"  Features: {available_features}")

if len(available_features) == 0:
    print("  ERROR: No operational features found!")
    print(f"  Available columns: {df.columns.tolist()}")
    exit()

# ============================================================================
# STEP 3: PREPARE DATA
# ============================================================================
print("\nSTEP 3: Preparing data...")

# Extract features and target
X = df[available_features].copy()
y = df[target].copy()

print(f"  Dataset shape: {X.shape}")
print(f"  Target range: {y.min():.2f} to {y.max():.2f}")

# Check for missing values
missing_before = X.isnull().sum().sum()
print(f"  Missing values: {missing_before}")

if missing_before > 0:
    print("  Handling missing values (filling with mean)...")
    X = X.fillna(X.mean())
    missing_after = X.isnull().sum().sum()
    print(f"  Missing values after: {missing_after}")

# Check how many complete cases we have
complete_cases = len(X.dropna())
print(f"  Complete cases: {complete_cases}/{len(X)}")

# ============================================================================
# STEP 4: SPLIT DATA (Train/Test)
# ============================================================================
print("\nSTEP 4: Splitting data into train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)

print(f"  Training set: {len(X_train)} earthquakes")
print(f"  Test set: {len(X_test)} earthquakes")

# ============================================================================
# STEP 5: TRAIN MODELS
# ============================================================================
print("\nSTEP 5: Training models...")
print("  This may take a minute...")

# Model 1: Linear Regression (baseline)
print("\n  Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Model 2: Random Forest (better performance)
print("  Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ============================================================================
# STEP 6: EVALUATE MODELS
# ============================================================================
print("\n" + "="*70)
print("STEP 6: MODEL PERFORMANCE")
print("="*70)

# Linear Regression metrics
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

print("\n1. LINEAR REGRESSION:")
print(f"   R² Score: {lr_r2:.3f}")
print(f"   Mean Absolute Error: {lr_mae:.3f}")
print(f"   Root Mean Squared Error: {lr_rmse:.3f}")

# Random Forest metrics
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\n2. RANDOM FOREST:")
print(f"   R² Score: {rf_r2:.3f}")
print(f"   Mean Absolute Error: {rf_mae:.3f}")
print(f"   Root Mean Squared Error: {rf_rmse:.3f}")

# Best model
if rf_r2 > lr_r2:
    print("\n   Winner: Random Forest! (better R² score)")
    best_model = rf_model
    best_pred = rf_pred
    best_name = "Random Forest"
else:
    print("\n   Winner: Linear Regression! (better R² score)")
    best_model = lr_model
    best_pred = lr_pred
    best_name = "Linear Regression"

# ============================================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("STEP 7: FEATURE IMPORTANCE")
print("="*70)
print("\nWhich operational parameters matter most?")
print("-"*70)

if best_name == "Random Forest":
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance (Random Forest):")
    for idx, row in importance_df.iterrows():
        bar = '█' * int(row['Importance'] * 100)
        print(f"  {row['Feature']:20} {row['Importance']:.3f} {bar}")
else:
    # Get coefficients for linear regression
    coef_df = pd.DataFrame({
        'Feature': available_features,
        'Coefficient': np.abs(lr_model.coef_)
    }).sort_values('Coefficient', ascending=False)
    
    print("\nFeature Importance (Linear Regression - Absolute Coefficients):")
    for idx, row in coef_df.iterrows():
        print(f"  {row['Feature']:20} {row['Coefficient']:.3f}")

# ============================================================================
# STEP 8: EXAMPLE PREDICTIONS
# ============================================================================
print("\n" + "="*70)
print("STEP 8: EXAMPLE PREDICTIONS")
print("="*70)

print("\nShowing 10 random test predictions:")
print("-"*70)
print(f"{'Actual':>10} {'Predicted':>10} {'Error':>10}")
print("-"*70)

# Show random 10 predictions
indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
for i in indices:
    actual = y_test.iloc[i]
    predicted = best_pred[i]
    error = abs(actual - predicted)
    print(f"{actual:>10.3f} {predicted:>10.3f} {error:>10.3f}")

# ============================================================================
# STEP 9: INTERPRETATION
# ============================================================================
print("\n" + "="*70)
print("STEP 9: WHAT DO THESE RESULTS MEAN?")
print("="*70)

print(f"""
R² Score: {rf_r2:.3f}
  - Ranges from 0 to 1 (higher is better)
  - {rf_r2:.1%} of variance in magnitude is explained by operational parameters
  - Interpretation: {'Excellent' if rf_r2 > 0.8 else 'Good' if rf_r2 > 0.6 else 'Moderate' if rf_r2 > 0.4 else 'Weak'} model

Mean Absolute Error: {rf_mae:.3f}
  - Average prediction error in magnitude units
  - On average, predictions are off by ±{rf_mae:.2f} magnitude
  - Interpretation: {'Very Accurate' if rf_mae < 0.2 else 'Good' if rf_mae < 0.4 else 'Moderate' if rf_mae < 0.6 else 'Needs Improvement'}

Conclusion:
  {' Model successfully links operations to earthquake magnitude!' if rf_r2 > 0.3 else ' Weak relationship - may need more features or data'}
  {' Predictions are reasonably accurate' if rf_mae < 0.5 else ' High prediction error - model needs improvement'}
""")

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("STEP 10: SAVING RESULTS")
print("="*70)

# Save predictions
results_df = pd.DataFrame({
    'actual_magnitude': y_test,
    'predicted_magnitude': best_pred,
    'error': abs(y_test - best_pred)
})
results_df.to_csv('ml_predictions.csv', index=False)
print("  Saved: ml_predictions.csv")

# Save model summary
with open('ml_model_summary.txt', 'w') as f:
    f.write("EARTHQUAKE MAGNITUDE PREDICTION MODEL SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model: {best_name}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"\nPerformance:\n")
    f.write(f"  R² Score: {rf_r2:.3f}\n")
    f.write(f"  Mean Absolute Error: {rf_mae:.3f}\n")
    f.write(f"  Root Mean Squared Error: {rf_rmse:.3f}\n")
    f.write(f"\nFeatures used:\n")
    for feat in available_features:
        f.write(f"  - {feat}\n")

print("  Saved: ml_model_summary.txt")

# Create simple visualization if matplotlib available
try:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title(f'{best_name}: Actual vs Predicted Magnitude\nR² = {rf_r2:.3f}, MAE = {rf_mae:.3f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ml_predictions_plot.png', dpi=150)
    print("  Saved: ml_predictions_plot.png")
    print("\n  Tip: Open ml_predictions_plot.png to see visualization!")
except Exception as e:
    print(f"  Note: Could not create plot ({e})")

# ============================================================================
# DONE!
# ============================================================================
print("\n" + "="*70)
print("DONE! MODEL TRAINING COMPLETE")
print("="*70)

print(f"""
Summary:
  - Trained {best_name} model
  - Tested on {len(X_test)} earthquakes
  - Achieved R2 = {rf_r2:.3f}, MAE = {rf_mae:.3f}
  - Saved results to ml_predictions.csv
  - Saved summary to ml_model_summary.txt

Next steps:
  1. Review ml_predictions.csv to see individual predictions
  2. Check ml_predictions_plot.png for visualization
  3. Analyze feature importance to understand key factors
  4. Try adding more features to improve performance
  5. Consider using XGBoost for potentially better results

Questions? Check COMPLETE_PROJECT_GUIDE.md for details!
""")

print("="*70)

