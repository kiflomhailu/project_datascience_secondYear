"""
Test the trained LSTM model with new data
"""

import numpy as np
from tensorflow.keras.models import load_model

print("Loading trained model...")
model = load_model('../models/lstm_simple_model.h5')

print("✓ Model loaded successfully!")
print(f"\nModel expects input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

print("\n" + "="*60)
print("MODEL READY FOR PREDICTIONS!")
print("="*60)
print("\nTo make predictions:")
print("1. Prepare 24 hours of operational data")
print("2. Scale using the same scaler from training")
print("3. Call: predictions = model.predict(new_data)")
print("4. Result: 0 = No event, 1 = Seismic event likely")
