"""
Lightweight Model Training - No TensorFlow Required!
Uses Random Forest & Logistic Regression (scikit-learn only)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib

print("="*60)
print("LIGHTWEIGHT SEISMIC RISK PREDICTION")
print("No TensorFlow needed - Using scikit-learn only!")
print("="*60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
seismic_df = pd.read_csv('../data/seismic_events.csv', parse_dates=['occurred_at'])
operational_df = pd.read_csv('../data/operational_metrics.csv', 
                             parse_dates=['recorded_at'], nrows=30000)

print(f"✓ Seismic: {len(seismic_df)} events")
print(f"✓ Operational: {len(operational_df)} rows")

# ============================================================================
# 2. PREPROCESS
# ============================================================================
print("\n[2/6] Preprocessing...")
# Clean operational data
operational_df = operational_df[['recorded_at', 'inj_flow', 'inj_whp', 
                                 'inj_temp', 'prod_temp', 'prod_whp']].fillna(0)
operational_df['recorded_at'] = operational_df['recorded_at'].dt.floor('H')

# Aggregate seismic by hour
seismic_df['hour'] = seismic_df['occurred_at'].dt.floor('H')
seismic_hourly = seismic_df.groupby('hour').agg({
    'magnitude': ['count', 'max', 'mean']
}).reset_index()
seismic_hourly.columns = ['recorded_at', 'event_count', 'max_mag', 'avg_mag']

# Merge
merged_df = pd.merge(operational_df, seismic_hourly, on='recorded_at', how='left')
merged_df[['event_count', 'max_mag', 'avg_mag']] = merged_df[['event_count', 'max_mag', 'avg_mag']].fillna(0)

print(f"✓ Merged: {len(merged_df)} rows")

# ============================================================================
# 3. CREATE TARGET (Traffic Light System)
# ============================================================================
print("\n[3/6] Creating target variable...")

# Calculate max magnitude in next 7 days (168 hours)
merged_df['future_max_mag'] = merged_df['max_mag'].rolling(168).max().shift(-168)

# Traffic light categories
def get_risk_level(mag):
    if mag < 0.5:
        return 0  # 🟢 Green
    elif mag < 1.0:
        return 1  # 🟡 Yellow
    elif mag < 1.5:
        return 2  # 🟠 Orange
    else:
        return 3  # 🔴 Red

merged_df['risk_level'] = merged_df['future_max_mag'].apply(get_risk_level)
merged_df = merged_df.dropna()

print("\nRisk Distribution:")
risk_counts = merged_df['risk_level'].value_counts().sort_index()
labels = ['🟢 Green', '🟡 Yellow', '🟠 Orange', '🔴 Red']
for i, label in enumerate(labels):
    count = risk_counts.get(i, 0)
    print(f"  {label}: {count} ({count/len(merged_df)*100:.1f}%)")

# ============================================================================
# 4. PREPARE FEATURES
# ============================================================================
print("\n[4/6] Preparing features...")
feature_cols = ['inj_flow', 'inj_whp', 'inj_temp', 'prod_temp', 'prod_whp',
                'event_count', 'max_mag', 'avg_mag']

X = merged_df[feature_cols]
y = merged_df['risk_level']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"✓ Train: {len(X_train)} samples")
print(f"✓ Test: {len(X_test)} samples")

# ============================================================================
# 5. TRAIN MODELS
# ============================================================================
print("\n[5/6] Training models...")

# Model 1: Random Forest (BEST for this task)
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"✓ Random Forest Accuracy: {rf_acc:.4f} ({rf_acc*100:.1f}%)")

# Model 2: Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"✓ Logistic Regression Accuracy: {lr_acc:.4f} ({lr_acc*100:.1f}%)")

# ============================================================================
# 6. EVALUATE & SAVE
# ============================================================================
print("\n[6/6] Evaluation & Saving...")

print("\n" + "="*60)
print("RANDOM FOREST - Classification Report:")
print("="*60)
print(classification_report(y_test, rf_pred, 
                          target_names=['🟢 Green', '🟡 Yellow', '🟠 Orange', '🔴 Red']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:<15} {row['importance']:.4f}")

# Save models
joblib.dump(rf_model, '../models/random_forest_model.pkl')
joblib.dump(lr_model, '../models/logistic_regression_model.pkl')
joblib.dump(scaler, '../models/scaler_light.pkl')

print("\n✓ Models saved:")
print("  - ../models/random_forest_model.pkl")
print("  - ../models/logistic_regression_model.pkl")
print("  - ../models/scaler_light.pkl")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Random Forest - Feature Importance')
plt.tight_layout()
plt.savefig('../models/feature_importance.png', dpi=150)
print("  - ../models/feature_importance.png")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\nBest Model: Random Forest with {rf_acc*100:.1f}% accuracy")
print("\nTo use:")
print("  import joblib")
print("  model = joblib.load('../models/random_forest_model.pkl')")
print("  predictions = model.predict(new_data)")
