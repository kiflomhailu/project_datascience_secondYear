"""
LSTM Model Training for Seismic Risk Prediction
Geothermal Energy - Traffic Light System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# Load seismic events
seismic_df = pd.read_csv('../data/seismic_events.csv')
print(f"Seismic Events: {seismic_df.shape}")
print(seismic_df.head())

# Load operational metrics (sample to reduce memory usage)
operational_df = pd.read_csv('../data/operational_metrics.csv', 
                             parse_dates=['recorded_at'],
                             nrows=100000)  # Adjust based on available memory
print(f"\nOperational Metrics: {operational_df.shape}")
print(operational_df.head())

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Convert seismic timestamps
seismic_df['occurred_at'] = pd.to_datetime(seismic_df['occurred_at'])

# Select relevant features from operational data
operational_features = ['recorded_at', 'inj_flow', 'inj_whp', 'inj_temp', 
                        'prod_temp', 'prod_whp', 'is_producing']
operational_df = operational_df[operational_features].copy()

# Handle missing values
operational_df = operational_df.fillna(method='ffill').fillna(0)

# Create time-based aggregations for seismic events
# Group by hour to match operational metrics
seismic_df['hour'] = seismic_df['occurred_at'].dt.floor('H')
seismic_hourly = seismic_df.groupby('hour').agg({
    'magnitude': ['count', 'max', 'mean'],
    'pgv_max': ['max', 'mean']
}).reset_index()

seismic_hourly.columns = ['recorded_at', 'event_count', 'max_magnitude', 
                          'avg_magnitude', 'max_pgv', 'avg_pgv']

# ============================================================================
# 3. MERGE DATA
# ============================================================================

print("\n" + "=" * 60)
print("MERGING DATASETS")
print("=" * 60)

# Round operational timestamps to hour
operational_df['recorded_at'] = pd.to_datetime(operational_df['recorded_at']).dt.floor('H')

# Merge datasets
merged_df = pd.merge(operational_df, seismic_hourly, 
                     on='recorded_at', how='left')

# Fill missing seismic data (no events in that hour)
seismic_cols = ['event_count', 'max_magnitude', 'avg_magnitude', 'max_pgv', 'avg_pgv']
merged_df[seismic_cols] = merged_df[seismic_cols].fillna(0)

print(f"Merged Dataset: {merged_df.shape}")
print(merged_df.head())

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Create target variable: Risk level for next 7 days (traffic light system)
# Green (0), Yellow (1), Orange (2), Red (3)

# Look ahead 7 days (168 hours)
lookback_hours = 24  # Use 24 hours of history
lookahead_hours = 168  # Predict 7 days ahead

# Calculate maximum magnitude in next 7 days
merged_df['future_max_magnitude'] = merged_df['max_magnitude'].rolling(
    window=lookahead_hours, min_periods=1).max().shift(-lookahead_hours)

# Create risk categories based on magnitude
def categorize_risk(magnitude):
    if magnitude < 0.5:
        return 0  # Green
    elif magnitude < 1.0:
        return 1  # Yellow
    elif magnitude < 1.5:
        return 2  # Orange
    else:
        return 3  # Red

merged_df['risk_level'] = merged_df['future_max_magnitude'].apply(categorize_risk)

# Drop rows with NaN in target
merged_df = merged_df.dropna(subset=['risk_level'])

print(f"Risk Level Distribution:")
print(merged_df['risk_level'].value_counts().sort_index())

# ============================================================================
# 5. PREPARE SEQUENCES FOR LSTM
# ============================================================================

print("\n" + "=" * 60)
print("PREPARING LSTM SEQUENCES")
print("=" * 60)

# Select features for LSTM
feature_cols = ['inj_flow', 'inj_whp', 'inj_temp', 'prod_temp', 'prod_whp',
                'event_count', 'max_magnitude', 'avg_magnitude', 'max_pgv', 'avg_pgv']

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_df[feature_cols])

# Create sequences
def create_sequences(data, targets, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, 
                        merged_df['risk_level'].values, 
                        lookback_hours)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ============================================================================
# 6. SPLIT DATA
# ============================================================================

print("\n" + "=" * 60)
print("SPLITTING DATA")
print("=" * 60)

# Split into train/validation/test (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# ============================================================================
# 7. BUILD LSTM MODEL
# ============================================================================

print("\n" + "=" * 60)
print("BUILDING LSTM MODEL")
print("=" * 60)

model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, 
         input_shape=(lookback_hours, len(feature_cols))),
    Dropout(0.2),
    
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.2),
    
    LSTM(32, activation='relu'),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes: Green, Yellow, Orange, Red
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# ============================================================================
# 8. TRAIN MODEL
# ============================================================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('../models/lstm_best_model.h5', 
                            monitor='val_accuracy', 
                            save_best_only=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ============================================================================
# 9. EVALUATE MODEL
# ============================================================================

print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

# Test set evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, 
                          target_names=['Green', 'Yellow', 'Orange', 'Red']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# ============================================================================
# 10. SAVE MODEL AND SCALER
# ============================================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

model.save('../models/lstm_seismic_risk_model.h5')
print("Model saved: ../models/lstm_seismic_risk_model.h5")

import joblib
joblib.dump(scaler, '../models/scaler.pkl')
print("Scaler saved: ../models/scaler.pkl")

# ============================================================================
# 11. PLOT TRAINING HISTORY
# ============================================================================

print("\n" + "=" * 60)
print("PLOTTING RESULTS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Model Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('../models/training_history.png', dpi=300, bbox_inches='tight')
print("Training history saved: ../models/training_history.png")

plt.show()

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
