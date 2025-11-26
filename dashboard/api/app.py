"""
============================================================================
FLASK API FOR LSTM SEISMIC RISK PREDICTION MODEL
============================================================================

Purpose: Backend API server that connects the trained LSTM model to the 
         React frontend dashboard. Provides endpoints for:
         - Health checks and model status
         - Operational data retrieval from CSV files
         - Seismic event data retrieval
         - Real-time risk predictions
         - 7-day forecast generation

Architecture:
- Flask REST API (port 5000)
- TensorFlow/Keras LSTM model (lstm_model_ammad.h5)
- StandardScaler for feature normalization
- CSV data loading with date filtering
- CORS enabled for frontend communication

Model Details:
- Input: 24 hours of operational data (lookback window)
- Features: 10 operational and seismic metrics
- Output: 4-class risk prediction (Green/Yellow/Orange/Red)
- Architecture: LSTM neural network for time-series forecasting

Author: Data Science Project - Hasselt University
Date: 2025
============================================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable cross-origin requests from frontend
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app with CORS support
app = Flask(__name__)
CORS(app)  # Allow frontend (localhost:8080) to call API (localhost:5000)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
model = None          # LSTM model instance (loaded at startup)
scaler = None         # StandardScaler for feature normalization
lookback_hours = 24   # LSTM requires 24 hours of historical data

# Feature columns - MUST match the features used during model training
# These are the 10 input features the LSTM model expects
FEATURE_COLS = [
    'inj_flow',        # Injection flow rate [m³/h]
    'inj_whp',        # Injection wellhead pressure [bar]
    'inj_temp',       # Injection temperature [°C]
    'prod_temp',      # Production temperature [°C]
    'prod_whp',       # Production wellhead pressure [bar]
    'event_count',    # Number of seismic events in time window
    'max_magnitude',  # Maximum seismic magnitude
    'avg_magnitude',  # Average seismic magnitude
    'max_pgv',        # Maximum peak ground velocity
    'avg_pgv'         # Average peak ground velocity
]

# Risk level mapping - corresponds to model output classes
RISK_LEVELS = ['Green', 'Yellow', 'Orange', 'Red']


# ============================================================================
# FEATURE PREPROCESSING FUNCTION
# ============================================================================
# Handles dimension mismatch between available features and model expectations
# Some models may expect more features than we have (e.g., 47 vs 10)
# This function pads with zeros or truncates as needed
# ============================================================================
def prepare_features_for_model(features_array, expected_features):
    """
    Prepare features array to match model's expected input size
    
    Args:
        features_array: Input features array (shape: batch, timesteps, features)
        expected_features: Number of features the model expects
    
    Returns:
        Adjusted features array matching model input shape
    """
    current_features = features_array.shape[-1]
    
    if current_features == expected_features:
        return features_array
    
    if current_features < expected_features:
        # Pad with zeros
        padding_size = expected_features - current_features
        padding = np.zeros((features_array.shape[0], features_array.shape[1], padding_size))
        return np.concatenate([features_array, padding], axis=-1)
    else:
        # Truncate (take first N features)
        return features_array[:, :, :expected_features]


def get_model_input_shape():
    """Get the expected input shape from the loaded model"""
    global model
    if model is None:
        return None
    try:
        # Get input shape: (batch, timesteps, features)
        input_shape = model.input_shape
        if input_shape:
            # Return (timesteps, features)
            return (input_shape[1], input_shape[2])
        return None
    except:
        return None

# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================
def load_model_and_scaler():
    """
    Load the trained LSTM model and StandardScaler from disk
    
    This function is called once at API startup to initialize the model.
    It searches multiple possible file paths to find the model and scaler files.
    
    Model File: lstm_model_ammad.h5 (Keras/TensorFlow HDF5 format)
    Scaler File: scaler.pkl (if available, otherwise creates dummy scaler)
    
    Returns: None (sets global variables model and scaler)
    """
    global model, scaler
    
    # Try different possible model paths (handles different directory structures)
    model_paths = [
        '../lstm_model_ammad.h5',
        '../models/lstm_best_model.h5',
        '../models/lstm_seismic_risk_model.h5',
        '../models/lstm_simple_model.h5',
        'lstm_model_ammad.h5',
        'models/lstm_best_model.h5',
        '../../dashboard/lstm_model_ammad.h5',
        '../../dashboard/models/lstm_best_model.h5'
    ]
    
    scaler_paths = [
        '../models/scaler.pkl',
        'models/scaler.pkl',
        'scaler.pkl',
        '../../dashboard/models/scaler.pkl'
    ]
    
    # Load model
    model_loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = load_model(path)
                print(f"[OK] Model loaded from: {path}")
                
                # Check model input shape
                try:
                    input_shape = model.input_shape
                    if input_shape:
                        expected_features = input_shape[2] if len(input_shape) > 2 else input_shape[1]
                        print(f"[INFO] Model expects {expected_features} features per timestep")
                        print(f"[INFO] Model input shape: {input_shape}")
                except Exception as e:
                    print(f"[WARNING] Could not determine model input shape: {e}")
                
                model_loaded = True
                break
            except Exception as e:
                print(f"[ERROR] Failed to load model from {path}: {e}")
    
    if not model_loaded:
        raise FileNotFoundError("Could not find model file. Please check the path.")
    
    # Load scaler
    scaler_loaded = False
    for path in scaler_paths:
        if os.path.exists(path):
            try:
                scaler = joblib.load(path)
                print(f"[OK] Scaler loaded from: {path}")
                scaler_loaded = True
                break
            except Exception as e:
                print(f"[ERROR] Failed to load scaler from {path}: {e}")
    
    if not scaler_loaded:
        print("[WARNING] Scaler not found. Creating dummy scaler (no scaling).")
        # Create a dummy scaler that doesn't scale (just returns data as-is)
        class DummyScaler:
            def transform(self, X):
                return np.array(X) if not isinstance(X, np.ndarray) else X
            def fit_transform(self, X):
                return np.array(X) if not isinstance(X, np.ndarray) else X
            def fit(self, X):
                return self
        scaler = DummyScaler()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint - verifies API and model status
    Returns: JSON with API status and model loading status
    Used by frontend to determine if real data can be loaded
    """
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint - predicts risk for one time point
    Expected JSON: {
        "data": [24 hours of operational data]
    }
    Returns: Risk level, probabilities, and confidence
    """
    """
    Predict seismic risk level from operational data
    
    Expected JSON body:
    {
        "data": [
            {
                "timestamp": "2024-01-01T00:00:00",
                "inj_flow": 0.5,
                "inj_whp": 0.8,
                "inj_temp": 150.0,
                "prod_temp": 120.0,
                "prod_whp": 0.6,
                "event_count": 0,
                "max_magnitude": 0.0,
                "avg_magnitude": 0.0,
                "max_pgv": 0.0,
                "avg_pgv": 0.0
            },
            ... (24 hours of data)
        ]
    }
    """
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400
        
        input_data = data['data']
        
        # Validate we have enough data points (need 24 hours)
        if len(input_data) < lookback_hours:
            return jsonify({
                'error': f'Need at least {lookback_hours} hours of data',
                'received': len(input_data)
            }), 400
        
        # Extract features in correct order
        features_list = []
        for record in input_data[-lookback_hours:]:  # Use last 24 hours
            feature_row = [record.get(col, 0.0) for col in FEATURE_COLS]
            features_list.append(feature_row)
        
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Reshape for LSTM: (1, lookback_hours, num_features)
        features_array = features_array.reshape(1, lookback_hours, len(FEATURE_COLS))
        
        # Scale features
        features_scaled = scaler.transform(features_array.reshape(-1, len(FEATURE_COLS)))
        features_scaled = features_scaled.reshape(1, lookback_hours, len(FEATURE_COLS))
        
        # Adjust features to match model's expected input size
        if model is not None:
            try:
                expected_features = model.input_shape[2] if model.input_shape and len(model.input_shape) > 2 else len(FEATURE_COLS)
                if expected_features != len(FEATURE_COLS):
                    print(f"[INFO] Adjusting features: {len(FEATURE_COLS)} → {expected_features}")
                    features_scaled = prepare_features_for_model(features_scaled, expected_features)
            except Exception as e:
                print(f"[WARNING] Could not adjust features: {e}")
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)
        
        # Handle different prediction shapes
        if len(prediction.shape) > 1 and prediction.shape[0] > 0:
            pred_array = prediction[0]
        else:
            pred_array = prediction.flatten()
        
        num_classes = len(pred_array)
        predicted_class = int(np.argmax(pred_array))
        probabilities = pred_array.tolist() if isinstance(pred_array, np.ndarray) else list(pred_array)
        
        # Map probabilities safely
        if num_classes >= 4:
            prob_dict = {
                'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                'yellow': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                'orange': float(probabilities[2]) if len(probabilities) > 2 else 0.0,
                'red': float(probabilities[3]) if len(probabilities) > 3 else 0.0
            }
            risk_level = RISK_LEVELS[predicted_class] if predicted_class < len(RISK_LEVELS) else 'Unknown'
        elif num_classes == 2:
            prob_dict = {
                'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                'yellow': 0.0,
                'orange': 0.0,
                'red': float(probabilities[1]) if len(probabilities) > 1 else 0.0
            }
            risk_level = 'Green' if predicted_class == 0 else 'Red'
        else:
            prob_dict = {
                'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                'yellow': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                'orange': float(probabilities[2]) if len(probabilities) > 2 else 0.0,
                'red': float(probabilities[3]) if len(probabilities) > 3 else 0.0
            }
            risk_level = RISK_LEVELS[min(predicted_class, len(RISK_LEVELS) - 1)]
        
        # Format response
        response = {
            'risk_level': risk_level,
            'risk_level_code': int(predicted_class),
            'probabilities': prob_dict,
            'confidence': float(max(probabilities)) if probabilities else 0.0
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict risk levels for multiple time periods
    
    Expected JSON body:
    {
        "data": [
            [24 hours of data for period 1],
            [24 hours of data for period 2],
            ...
        ]
    }
    """
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400
        
        batch_data = data['data']
        predictions = []
        
        for period_data in batch_data:
            if len(period_data) < lookback_hours:
                continue
            
            # Extract features
            features_list = []
            for record in period_data[-lookback_hours:]:
                feature_row = [record.get(col, 0.0) for col in FEATURE_COLS]
                features_list.append(feature_row)
            
            features_array = np.array(features_list)
            features_array = features_array.reshape(1, lookback_hours, len(FEATURE_COLS))
            
            # Scale
            features_scaled = scaler.transform(features_array.reshape(-1, len(FEATURE_COLS)))
            features_scaled = features_scaled.reshape(1, lookback_hours, len(FEATURE_COLS))
            
            # Adjust features to match model's expected input size
            if model is not None:
                try:
                    expected_features = model.input_shape[2] if model.input_shape and len(model.input_shape) > 2 else len(FEATURE_COLS)
                    if expected_features != len(FEATURE_COLS):
                        features_scaled = prepare_features_for_model(features_scaled, expected_features)
                except Exception as e:
                    print(f"[WARNING] Could not adjust features: {e}")
            
            # Predict
            prediction = model.predict(features_scaled, verbose=0)
            
            # Handle different prediction shapes
            if len(prediction.shape) > 1 and prediction.shape[0] > 0:
                pred_array = prediction[0]
            else:
                pred_array = prediction.flatten()
            
            num_classes = len(pred_array)
            predicted_class = int(np.argmax(pred_array))
            probabilities = pred_array.tolist() if isinstance(pred_array, np.ndarray) else list(pred_array)
            
            # Map probabilities safely
            if num_classes >= 4:
                prob_dict = {
                    'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                    'yellow': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                    'orange': float(probabilities[2]) if len(probabilities) > 2 else 0.0,
                    'red': float(probabilities[3]) if len(probabilities) > 3 else 0.0
                }
                risk_level = RISK_LEVELS[predicted_class] if predicted_class < len(RISK_LEVELS) else 'Unknown'
            elif num_classes == 2:
                prob_dict = {
                    'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                    'yellow': 0.0,
                    'orange': 0.0,
                    'red': float(probabilities[1]) if len(probabilities) > 1 else 0.0
                }
                risk_level = 'Green' if predicted_class == 0 else 'Red'
            else:
                prob_dict = {
                    'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                    'yellow': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                    'orange': float(probabilities[2]) if len(probabilities) > 2 else 0.0,
                    'red': float(probabilities[3]) if len(probabilities) > 3 else 0.0
                }
                risk_level = RISK_LEVELS[min(predicted_class, len(RISK_LEVELS) - 1)]
            
            predictions.append({
                'risk_level': risk_level,
                'risk_level_code': int(predicted_class),
                'probabilities': prob_dict
            })
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/data/operational', methods=['GET'])
def get_operational_data():
    """
    Get operational metrics data for specified time range
    Loads data from CSV files and filters by date range
    
    Query Parameters:
        start_date: Start date/time (ISO format, e.g., "2018-11-28T00:00:00")
        end_date: End date/time (ISO format)
        limit: Maximum number of records to return (default: 1000)
    
    Returns: JSON with array of operational records including:
        - timestamp, inj_flow, inj_whp, inj_temp, prod_temp, prod_whp
    
    Data Source: operational_metrics.csv (loaded from data/ folder)
    """
    try:
        import pandas as pd
        import os
        from datetime import datetime
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = int(request.args.get('limit', 1000))  # Default 1000 records
        
        # Try to load operational data
        operational_paths = [
            '../data/operational_metrics.csv',
            '../dashboard/data/operational_metrics.csv',
            'data/operational_metrics.csv'
        ]
        
        operational_df = None
        for path in operational_paths:
            if os.path.exists(path):
                try:
                    # Read full CSV first (or sample if very large)
                    # For date filtering, we need to read the full file or at least check date ranges
                    operational_df = pd.read_csv(path, nrows=100000)  # Read up to 100k rows for date filtering
                    
                    # Parse dates
                    if 'recorded_at' in operational_df.columns:
                        operational_df['recorded_at'] = pd.to_datetime(operational_df['recorded_at'], errors='coerce')
                    
                    # Filter by date range if provided
                    if start_date and end_date:
                        start = pd.to_datetime(start_date)
                        end = pd.to_datetime(end_date)
                        
                        # Filter by date range FIRST
                        operational_df = operational_df[
                            (operational_df['recorded_at'] >= start) & 
                            (operational_df['recorded_at'] <= end)
                        ]
                        
                        # If no data in range, return empty instead of fallback
                        if len(operational_df) == 0:
                            print(f"[WARN] No data in range {start} to {end}")
                            operational_df = pd.DataFrame()  # Empty dataframe
                        else:
                            # Limit results after filtering
                            if len(operational_df) > limit:
                                operational_df = operational_df.head(limit)
                            print(f"[OK] Filtered to {len(operational_df)} records in date range {start} to {end}")
                    else:
                        # No date filter - just limit
                        if len(operational_df) > limit:
                            operational_df = operational_df.tail(limit)  # Get latest if no date filter
                    
                    print(f"[OK] Loaded {len(operational_df)} operational records")
                    break
                except Exception as e:
                    print(f"[ERROR] Failed to load from {path}: {e}")
                    continue
        
        if operational_df is None or len(operational_df) == 0:
            return jsonify({'error': 'No operational data found'}), 404
        
        # Convert to JSON format
        result = []
        for _, row in operational_df.iterrows():
            record = {
                'timestamp': row['recorded_at'].isoformat() if pd.notna(row.get('recorded_at')) else None,
                'inj_flow': float(row.get('inj_flow', 0)) if pd.notna(row.get('inj_flow')) else 0,
                'inj_whp': float(row.get('inj_whp', 0)) if pd.notna(row.get('inj_whp')) else 0,
                'inj_temp': float(row.get('inj_temp', 0)) if pd.notna(row.get('inj_temp')) else 0,
                'prod_temp': float(row.get('prod_temp', 0)) if pd.notna(row.get('prod_temp')) else 0,
                'prod_whp': float(row.get('prod_whp', 0)) if pd.notna(row.get('prod_whp')) else 0,
            }
            result.append(record)
        
        return jsonify({
            'data': result,
            'count': len(result),
            'source': 'real_csv'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/data/seismic', methods=['GET'])
def get_seismic_data():
    """
    Get seismic events data for specified time range
    Loads seismic events from CSV and filters by date range
    
    Query Parameters:
        start_date: Start date/time (ISO format)
        end_date: End date/time (ISO format)
    
    Returns: JSON with array of seismic event records including:
        - timestamp, magnitude, pgv_max, x, y, z coordinates
    
    Data Source: seismic_events.csv (loaded from data/ folder)
    Returns empty array if no events found (not an error)
    """
    try:
        import pandas as pd
        import os
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Try to load seismic data
        seismic_paths = [
            '../data/seismic_events.csv',
            '../dashboard/data/seismic_events.csv',
            'data/seismic_events.csv'
        ]
        
        seismic_df = None
        for path in seismic_paths:
            if os.path.exists(path):
                try:
                    seismic_df = pd.read_csv(path)
                    
                    # Parse dates
                    if 'occurred_at' in seismic_df.columns:
                        seismic_df['occurred_at'] = pd.to_datetime(seismic_df['occurred_at'], errors='coerce')
                    
                    # Filter by date range if provided
                    if start_date and end_date:
                        start = pd.to_datetime(start_date)
                        end = pd.to_datetime(end_date)
                        seismic_df = seismic_df[
                            (seismic_df['occurred_at'] >= start) & 
                            (seismic_df['occurred_at'] <= end)
                        ]
                    
                    print(f"[OK] Loaded {len(seismic_df)} seismic records")
                    break
                except Exception as e:
                    print(f"[ERROR] Failed to load from {path}: {e}")
                    continue
        
        if seismic_df is None or len(seismic_df) == 0:
            # Return empty array instead of 404 - allows frontend to continue
            print(f"[WARN] No seismic data found for range {start_date} to {end_date}")
            return jsonify({
                'data': [],
                'count': 0,
                'source': 'real_csv',
                'message': 'No seismic data found for this date range'
            })
        
        # Convert to JSON format
        result = []
        for _, row in seismic_df.iterrows():
            record = {
                'timestamp': row['occurred_at'].isoformat() if pd.notna(row.get('occurred_at')) else None,
                'magnitude': float(row.get('magnitude', 0)) if pd.notna(row.get('magnitude')) else 0,
                'pgv_max': float(row.get('pgv_max', 0)) if pd.notna(row.get('pgv_max')) else 0,
                'x': float(row.get('x', 0)) if pd.notna(row.get('x')) else 0,
                'y': float(row.get('y', 0)) if pd.notna(row.get('y')) else 0,
                'z': float(row.get('z', 0)) if pd.notna(row.get('z')) else 0,
            }
            result.append(record)
        
        return jsonify({
            'data': result,
            'count': len(result),
            'source': 'real_csv'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/data/latest', methods=['GET'])
def get_latest_data():
    """
    Get the latest 24 hours of operational and seismic data
    Loads from CSV files on the server side
    """
    try:
        import pandas as pd
        import os
        
        # Try to load operational data
        operational_paths = [
            '../data/operational_metrics.csv',
            '../dashboard/data/operational_metrics.csv',
            'data/operational_metrics.csv'
        ]
        
        operational_df = None
        for path in operational_paths:
            if os.path.exists(path):
                try:
                    # Read last 1000 rows efficiently
                    # Count total rows first (quick check)
                    with open(path, 'r') as f:
                        total_rows = sum(1 for _ in f) - 1  # Subtract header
                    
                    if total_rows > 1000:
                        # Skip to last 1000 rows
                        skip_rows = total_rows - 1000
                        operational_df = pd.read_csv(path, skiprows=range(1, skip_rows + 1))
                    else:
                        operational_df = pd.read_csv(path)
                    
                    print(f"[OK] Loaded {len(operational_df)} operational records from {path}")
                    break
                except Exception as e:
                    print(f"[ERROR] Failed to load from {path}: {e}")
                    continue
        
        # Try to load seismic data
        seismic_paths = [
            '../data/seismic_events.csv',
            '../dashboard/data/seismic_events.csv',
            'data/seismic_events.csv'
        ]
        
        seismic_df = None
        for path in seismic_paths:
            if os.path.exists(path):
                try:
                    seismic_df = pd.read_csv(path)
                    print(f"[OK] Loaded {len(seismic_df)} seismic records from {path}")
                    break
                except Exception as e:
                    print(f"[ERROR] Failed to load from {path}: {e}")
                    continue
        
        if operational_df is None or len(operational_df) == 0:
            return jsonify({'error': 'No operational data found'}), 404
        
        # Get last 24 records
        recent_ops = operational_df.tail(24).copy()
        
        # Convert timestamps
        if 'recorded_at' in recent_ops.columns:
            recent_ops['recorded_at'] = pd.to_datetime(recent_ops['recorded_at'], errors='coerce')
        
        # Group seismic events by hour if available
        seismic_by_hour = {}
        if seismic_df is not None and 'occurred_at' in seismic_df.columns:
            seismic_df['occurred_at'] = pd.to_datetime(seismic_df['occurred_at'], errors='coerce')
            seismic_df['hour'] = seismic_df['occurred_at'].dt.floor('H')
            for hour, group in seismic_df.groupby('hour'):
                hour_str = hour.isoformat() if pd.notna(hour) else None
                if hour_str:
                    seismic_by_hour[hour_str] = {
                        'count': len(group),
                        'magnitudes': group['magnitude'].dropna().tolist() if 'magnitude' in group.columns else [],
                        'pgvs': group['pgv_max'].dropna().tolist() if 'pgv_max' in group.columns else []
                    }
        
        # Format data for API
        formatted_data = []
        for _, row in recent_ops.iterrows():
            timestamp = row.get('recorded_at', pd.Timestamp.now())
            if pd.notna(timestamp):
                hour_str = pd.Timestamp(timestamp).floor('H').isoformat()
            else:
                hour_str = None
            
            seismic = seismic_by_hour.get(hour_str, {}) if hour_str else {}
            
            record = {
                'timestamp': timestamp.isoformat() if pd.notna(timestamp) else pd.Timestamp.now().isoformat(),
                'inj_flow': float(row.get('inj_flow', 0)) if pd.notna(row.get('inj_flow')) else 0,
                'inj_whp': float(row.get('inj_whp', 0)) if pd.notna(row.get('inj_whp')) else 0,
                'inj_temp': float(row.get('inj_temp', 0)) if pd.notna(row.get('inj_temp')) else 0,
                'prod_temp': float(row.get('prod_temp', 0)) if pd.notna(row.get('prod_temp')) else 0,
                'prod_whp': float(row.get('prod_whp', 0)) if pd.notna(row.get('prod_whp')) else 0,
                'event_count': seismic.get('count', 0),
                'max_magnitude': max(seismic.get('magnitudes', [])) if seismic.get('magnitudes') else 0,
                'avg_magnitude': sum(seismic.get('magnitudes', [])) / len(seismic.get('magnitudes', [])) if seismic.get('magnitudes') else 0,
                'max_pgv': max(seismic.get('pgvs', [])) if seismic.get('pgvs') else 0,
                'avg_pgv': sum(seismic.get('pgvs', [])) / len(seismic.get('pgvs', [])) if seismic.get('pgvs') else 0,
            }
            formatted_data.append(record)
        
        return jsonify({
            'data': formatted_data,
            'source': 'real_csv',
            'records_loaded': len(formatted_data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/forecast', methods=['POST'])
def predict_forecast():
    """
    Generate 7-day seismic risk forecast using LSTM model
    
    This is the main prediction endpoint used by the Risk Dashboard.
    Takes the last 24 hours of operational data and generates
    risk predictions for the next 7 days.
    
    Expected JSON body:
    {
        "start_date": "2024-01-01T00:00:00",  # Starting date for forecast
        "historical_data": [                   # Last 24 hours of data (required)
            {timestamp, inj_flow, inj_whp, ...},  # Hour 1
            {timestamp, inj_flow, inj_whp, ...},  # Hour 2
            ...                                    # ... 24 hours total
        ]
    }
    
    Returns: JSON with 7-day forecast array:
    {
        "forecast": [
            {
                "date": "2024-01-01",
                "risk_level": "Green",
                "risk_level_code": 0,
                "probabilities": {
                    "green": 0.95,
                    "yellow": 0.03,
                    "orange": 0.01,
                    "red": 0.01
                }
            },
            ...  # 7 days total
        ]
    }
    
    Model Process:
    1. Takes last 24 hours of historical data
    2. Scales features using StandardScaler
    3. Reshapes to LSTM input format: (1, 24, features)
    4. Runs prediction through LSTM model
    5. Applies softmax to convert logits to probabilities
    6. Maps to risk levels (Green/Yellow/Orange/Red)
    7. Repeats for each of 7 forecast days
    """
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        historical = data.get('historical_data', [])
        
        if len(historical) < lookback_hours:
            return jsonify({'error': f'Need at least {lookback_hours} hours of historical data'}), 400
        
        # For now, return predictions based on historical data
        # In production, you'd use future operational data if provided
        forecast = []
        start_date = datetime.fromisoformat(data.get('start_date', datetime.now().isoformat()))
        
        for day in range(7):
            # Use last 24 hours for prediction
            features_list = []
            for record in historical[-lookback_hours:]:
                feature_row = [record.get(col, 0.0) for col in FEATURE_COLS]
                features_list.append(feature_row)
            
            features_array = np.array(features_list)
            features_array = features_array.reshape(1, lookback_hours, len(FEATURE_COLS))
            
            # Scale
            features_scaled = scaler.transform(features_array.reshape(-1, len(FEATURE_COLS)))
            features_scaled = features_scaled.reshape(1, lookback_hours, len(FEATURE_COLS))
            
            # Adjust features to match model's expected input size
            if model is not None:
                try:
                    expected_features = model.input_shape[2] if model.input_shape and len(model.input_shape) > 2 else len(FEATURE_COLS)
                    if expected_features != len(FEATURE_COLS):
                        features_scaled = prepare_features_for_model(features_scaled, expected_features)
                except Exception as e:
                    print(f"[WARNING] Could not adjust features: {e}")
            
            # Predict
            prediction = model.predict(features_scaled, verbose=0)
            
            # Handle different prediction shapes
            if len(prediction.shape) > 1 and prediction.shape[0] > 0:
                pred_array = prediction[0]
            else:
                pred_array = prediction.flatten()
            
            # Check prediction shape
            num_classes = len(pred_array)
            print(f"[DEBUG] Prediction shape: {prediction.shape}, num_classes: {num_classes}, raw values: {pred_array}")
            
            # Apply softmax if values don't sum to ~1.0 (might be logits)
            pred_sum = np.sum(pred_array)
            if abs(pred_sum - 1.0) > 0.1:  # If not already probabilities, apply softmax
                print(f"[DEBUG] Applying softmax (sum={pred_sum:.4f})")
                exp_pred = np.exp(pred_array - np.max(pred_array))  # Subtract max for numerical stability
                probabilities = (exp_pred / np.sum(exp_pred)).tolist()
            else:
                probabilities = pred_array.tolist() if isinstance(pred_array, np.ndarray) else list(pred_array)
            
            predicted_class = int(np.argmax(probabilities))
            print(f"[DEBUG] Probabilities after processing: {probabilities}, predicted_class: {predicted_class}")
            
            # Map probabilities to risk levels (handle different number of classes)
            if num_classes == 4:
                prob_dict = {
                    'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                    'yellow': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                    'orange': float(probabilities[2]) if len(probabilities) > 2 else 0.0,
                    'red': float(probabilities[3]) if len(probabilities) > 3 else 0.0
                }
                risk_level = RISK_LEVELS[predicted_class] if predicted_class < len(RISK_LEVELS) else 'Unknown'
                print(f"[FORECAST] Day {day+1} - Green: {prob_dict['green']:.4f}, Yellow: {prob_dict['yellow']:.4f}, Orange: {prob_dict['orange']:.4f}, Red: {prob_dict['red']:.4f}")
            elif num_classes == 2:
                # Binary classification
                prob_dict = {
                    'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                    'yellow': 0.0,
                    'orange': 0.0,
                    'red': float(probabilities[1]) if len(probabilities) > 1 else 0.0
                }
                risk_level = 'Green' if predicted_class == 0 else 'Red'
            else:
                # Unknown number of classes - use first 4 or pad
                prob_dict = {
                    'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                    'yellow': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                    'orange': float(probabilities[2]) if len(probabilities) > 2 else 0.0,
                    'red': float(probabilities[3]) if len(probabilities) > 3 else 0.0
                }
                risk_level = RISK_LEVELS[min(predicted_class, len(RISK_LEVELS) - 1)]
            
            forecast.append({
                'date': (start_date + timedelta(days=day)).isoformat(),
                'risk_level': risk_level,
                'risk_level_code': int(predicted_class),
                'probabilities': prob_dict
            })
        
        return jsonify({'forecast': forecast})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Loading LSTM Model...")
    print("=" * 60)
    
    try:
        load_model_and_scaler()
        print("\n" + "=" * 60)
        print("API Server Starting...")
        print("=" * 60)
        print("Endpoints:")
        print("  GET  /health - Health check")
        print("  POST /predict - Single prediction")
        print("  POST /predict/batch - Batch predictions")
        print("  POST /predict/forecast - 7-day forecast")
        print("\nServer running on http://localhost:5000")
        print("=" * 60)
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"\n[ERROR] Error starting server: {e}")
        print("Please check that the model file exists and is accessible.")

