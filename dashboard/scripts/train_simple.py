"""
Quick Start: LSTM Training Script
Simplified version for initial testing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

print("Loading data...")
# Load seismic events
seismic_df = pd.read_csv('../data/seismic_events.csv', parse_dates=['occurred_at'])
print(f"✓ Seismic events: {len(seismic_df)} rows")

# Load operational metrics (first 50k rows for speed)
operational_df = pd.read_csv('../data/operational_metrics.csv', 
                             parse_dates=['recorded_at'],
                             nrows=50000)
print(f"✓ Operational metrics: {len(operational_df)} rows")

print("\nPreprocessing data...")
# Prepare operational features
operational_df = operational_df[['recorded_at', 'inj_flow', 'inj_whp', 'inj_temp']].fillna(0)
operational_df['recorded_at'] = operational_df['recorded_at'].dt.floor('H')

# Aggregate seismic events by hour
seismic_df['hour'] = seismic_df['occurred_at'].dt.floor('H')
seismic_hourly = seismic_df.groupby('hour').agg({
    'magnitude': ['count', 'max', 'mean']
}).reset_index()
seismic_hourly.columns = ['recorded_at', 'event_count', 'max_magnitude', 'avg_magnitude']

# Merge
merged_df = pd.merge(operational_df, seismic_hourly, on='recorded_at', how='left')
merged_df[['event_count', 'max_magnitude', 'avg_magnitude']] = merged_df[['event_count', 'max_magnitude', 'avg_magnitude']].fillna(0)

print(f"✓ Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")

# Create target: Will there be a seismic event in next 24 hours?
merged_df['target'] = (merged_df['event_count'].rolling(24).sum().shift(-24) > 0).astype(int)
merged_df = merged_df.dropna()

print(f"\nTarget distribution:")
print(merged_df['target'].value_counts())

# Prepare features
features = ['inj_flow', 'inj_whp', 'inj_temp', 'event_count', 'max_magnitude']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_df[features])

# Create sequences
LOOKBACK = 24
X, y = [], []
for i in range(LOOKBACK, len(scaled_data)):
    X.append(scaled_data[i-LOOKBACK:i])
    y.append(merged_df['target'].iloc[i])

X = np.array(X)
y = np.array(y)

print(f"\nSequence shape: X={X.shape}, y={y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Build model
print("\nBuilding LSTM model...")
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(LOOKBACK, len(features))),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train
print("\n" + "="*60)
print("TRAINING...")
print("="*60)
history = model.fit(X_train, y_train, 
                   validation_split=0.2,
                   epochs=20,
                   batch_size=32,
                   verbose=1)

# Evaluate
print("\n" + "="*60)
print("EVALUATION")
print("="*60)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save
model.save('../models/lstm_simple_model.h5')
print("\n✓ Model saved: ../models/lstm_simple_model.h5")

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('../models/simple_training_history.png', dpi=150)
print("✓ Plot saved: ../models/simple_training_history.png")

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)
