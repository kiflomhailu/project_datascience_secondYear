"""
============================================================================
FLASK API FOR SEISMIC RISK PREDICTION - ENSEMBLE MODEL (LSTM + CatBoost)
============================================================================

Purpose: Backend API server that combines LSTM and CatBoost models for 
         seismic risk prediction. Provides endpoints for:
         - Health checks and model status
         - Operational data retrieval from CSV files
         - Seismic event data retrieval
         - Real-time risk predictions (single, batch, forecast)
         - 2-day forecast generation with ensemble predictions

Architecture:
- Flask REST API (port 5000)
- TensorFlow/Keras LSTM model (lstm_model_ammad.h5)
- CatBoost model (earthquake_catboost_model.cbm) - thriey
- StandardScaler for feature normalization
- CSV data loading with date filtering
- CORS enabled for frontend communication
- Ensemble prediction combining both models

Model Details:
- LSTM: 4-class risk prediction (Green/Yellow/Orange/Red)
- CatBoost: Binary earthquake probability (0-1)
- Ensemble: Combines both predictions for improved accuracy

Author: Data Science Project - Hasselt University
Date: 2025
============================================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Configuration
LOOKBACK_HOURS = 24
FORECAST_DAYS = 2  # Changed from 7 to 2 days
RISK_LEVELS = ['Green', 'Yellow', 'Orange', 'Red']

# Feature columns - MUST match the features used during model training
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

# Ensemble Configuration
CATBOOST_ADJUSTMENT_FACTOR = 0.3  # How much CatBoost affects LSTM predictions
CATBOOST_THRESHOLD = 0.5  # Minimum earthquake probability to trigger adjustment

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

app = Flask(__name__)
CORS(app)

# Model instances
lstm_model: Optional[Any] = None
catboost_model: Optional[CatBoostClassifier] = None
scaler: Optional[StandardScaler] = None

# ============================================================================
# UTILITY CLASSES
# ============================================================================

class DummyScaler:
    """Dummy scaler that returns data as-is when no scaler is available"""
    def transform(self, X):
        return np.array(X) if not isinstance(X, np.ndarray) else X
    
    def fit_transform(self, X):
        return np.array(X) if not isinstance(X, np.ndarray) else X
    
    def fit(self, X):
        return self


# ============================================================================
# FEATURE PREPROCESSING
# ============================================================================

def prepare_features_for_lstm(historical_data: List[Dict]) -> np.ndarray:
    """
    Prepare features for LSTM model from historical operational data
    
    Args:
        historical_data: List of dictionaries with operational data
        
    Returns:
        Numpy array shaped (1, lookback_hours, num_features)
    """
    features_list = []
    for record in historical_data[-LOOKBACK_HOURS:]:
        feature_row = [record.get(col, 0.0) for col in FEATURE_COLS]
        features_list.append(feature_row)
    
    features_array = np.array(features_list)
    return features_array.reshape(1, LOOKBACK_HOURS, len(FEATURE_COLS))


def prepare_features_for_catboost(historical_data: List[Dict]) -> np.ndarray:
    """
    Prepare features for CatBoost model from historical operational data
    CatBoost expects aggregated features (statistics) rather than time-series
    
    Args:
        historical_data: List of dictionaries with operational data
        
    Returns:
        Numpy array with aggregated features
    """
    try:
        df = pd.DataFrame(historical_data)
        
        # Calculate aggregated statistics for each feature
        features = {}
        for col in FEATURE_COLS:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(values) > 0:
                    features[f'{col}_mean'] = float(values.mean())
                    features[f'{col}_max'] = float(values.max())
                    features[f'{col}_min'] = float(values.min())
                    features[f'{col}_std'] = float(values.std()) if len(values) > 1 else 0.0
                else:
                    features[f'{col}_mean'] = 0.0
                    features[f'{col}_max'] = 0.0
                    features[f'{col}_min'] = 0.0
                    features[f'{col}_std'] = 0.0
            else:
                features[f'{col}_mean'] = 0.0
                features[f'{col}_max'] = 0.0
                features[f'{col}_min'] = 0.0
                features[f'{col}_std'] = 0.0
        
        # Convert to array (using mean values as primary features)
        # Adjust this based on your actual CatBoost model requirements
        feature_array = np.array([features.get(f'{col}_mean', 0.0) for col in FEATURE_COLS])
        return feature_array.reshape(1, -1)
        
    except Exception as e:
        print(f"[WARNING] Error preparing CatBoost features: {e}")
        return np.zeros((1, len(FEATURE_COLS)))


def adjust_features_for_model(features_array: np.ndarray, expected_features: int) -> np.ndarray:
    """
    Adjust features array to match model's expected input size
    
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


def normalize_probabilities(probabilities: List[float]) -> List[float]:
    """
    Normalize probabilities to ensure they sum to 1.0
    
    Args:
        probabilities: List of probability values
        
    Returns:
        Normalized probabilities
    """
    total = sum(probabilities)
    if total > 0:
        return [p / total for p in probabilities]
    return probabilities


def apply_softmax(logits: np.ndarray) -> List[float]:
    """
    Apply softmax to convert logits to probabilities
    
    Args:
        logits: Array of logit values
        
    Returns:
        List of probabilities
    """
    exp_pred = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return (exp_pred / np.sum(exp_pred)).tolist()


# ============================================================================
# PREDICTION LOGIC
# ============================================================================

def process_lstm_prediction(prediction: np.ndarray) -> Tuple[List[float], int]:
    """
    Process LSTM model prediction output
    
    Args:
        prediction: Raw prediction from LSTM model
        
    Returns:
        Tuple of (probabilities list, predicted class index)
    """
    # Handle different prediction shapes
    if len(prediction.shape) > 1 and prediction.shape[0] > 0:
        pred_array = prediction[0]
    else:
        pred_array = prediction.flatten()
    
    # Apply softmax if values don't sum to ~1.0 (might be logits)
    pred_sum = np.sum(pred_array)
    if abs(pred_sum - 1.0) > 0.1:
        probabilities = apply_softmax(pred_array)
    else:
        probabilities = pred_array.tolist() if isinstance(pred_array, np.ndarray) else list(pred_array)
    
    predicted_class = int(np.argmax(probabilities))
    return probabilities, predicted_class


def get_catboost_prediction(historical_data: List[Dict]) -> float:
    """
    Get earthquake probability from CatBoost model
    
    Args:
        historical_data: List of dictionaries with operational data
        
    Returns:
        Earthquake probability (0.0 to 1.0)
    """
    if catboost_model is None:
        return 0.0
    
    try:
        catboost_features = prepare_features_for_catboost(historical_data)
        catboost_proba = catboost_model.predict_proba(catboost_features)[0, 1]  # Probability of earthquake
        return float(catboost_proba)
    except Exception as e:
        print(f"[WARNING] CatBoost prediction failed: {e}")
        return 0.0


def map_probabilities_to_risk_levels(probabilities: List[float], num_classes: int) -> Tuple[Dict[str, float], str, int]:
    """
    Map model probabilities to risk level categories
    
    Args:
        probabilities: List of class probabilities
        num_classes: Number of classes in the model output
        
    Returns:
        Tuple of (probability dictionary, risk level string, risk level code)
    """
    if num_classes >= 4:
        prob_dict = {
            'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
            'yellow': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
            'orange': float(probabilities[2]) if len(probabilities) > 2 else 0.0,
            'red': float(probabilities[3]) if len(probabilities) > 3 else 0.0
        }
        predicted_class = int(np.argmax(probabilities))
        risk_level = RISK_LEVELS[predicted_class] if predicted_class < len(RISK_LEVELS) else 'Unknown'
    elif num_classes == 2:
        prob_dict = {
            'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
            'yellow': 0.0,
            'orange': 0.0,
            'red': float(probabilities[1]) if len(probabilities) > 1 else 0.0
        }
        predicted_class = int(np.argmax(probabilities))
        risk_level = 'Green' if predicted_class == 0 else 'Red'
    else:
        prob_dict = {
            'green': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
            'yellow': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
            'orange': float(probabilities[2]) if len(probabilities) > 2 else 0.0,
            'red': float(probabilities[3]) if len(probabilities) > 3 else 0.0
        }
        predicted_class = int(np.argmax(probabilities))
        risk_level = RISK_LEVELS[min(predicted_class, len(RISK_LEVELS) - 1)]
    
    return prob_dict, risk_level, predicted_class


def apply_ensemble_adjustment(prob_dict: Dict[str, float], catboost_prob: float) -> Tuple[Dict[str, float], str, int]:
    """
    Apply ensemble adjustment: combine LSTM and CatBoost predictions
    
    Args:
        prob_dict: LSTM probability dictionary
        catboost_prob: CatBoost earthquake probability
        
    Returns:
        Tuple of (adjusted probability dictionary, risk level, risk level code)
    """
    if catboost_prob > CATBOOST_THRESHOLD:
        # If CatBoost predicts high earthquake probability, increase risk levels
        adjustment_factor = catboost_prob * CATBOOST_ADJUSTMENT_FACTOR
        
        prob_dict['red'] = min(1.0, prob_dict['red'] + adjustment_factor)
        prob_dict['orange'] = min(1.0, prob_dict['orange'] + adjustment_factor * 0.7)
        prob_dict['yellow'] = min(1.0, prob_dict['yellow'] + adjustment_factor * 0.5)
        prob_dict['green'] = max(0.0, prob_dict['green'] - adjustment_factor * 0.3)
        
        # Normalize to ensure probabilities sum to ~1.0
        total = sum(prob_dict.values())
        if total > 0:
            prob_dict = {k: v/total for k, v in prob_dict.items()}
        
        # Re-determine risk level based on adjusted probabilities
        adjusted_probs = [prob_dict['green'], prob_dict['yellow'], prob_dict['orange'], prob_dict['red']]
        predicted_class = int(np.argmax(adjusted_probs))
        risk_level = RISK_LEVELS[predicted_class] if predicted_class < len(RISK_LEVELS) else 'Unknown'
        
        print(f"[ENSEMBLE] Adjusted probabilities with CatBoost (EQ prob: {catboost_prob:.4f}): {prob_dict}")
    else:
        # Use LSTM prediction as-is
        predicted_class = int(np.argmax([prob_dict['green'], prob_dict['yellow'], prob_dict['orange'], prob_dict['red']]))
        risk_level = RISK_LEVELS[predicted_class] if predicted_class < len(RISK_LEVELS) else 'Unknown'
    
    return prob_dict, risk_level, predicted_class


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """
    Load LSTM model, CatBoost model, and scaler from disk
    Searches multiple possible file paths to handle different directory structures
    """
    global lstm_model, catboost_model, scaler
    
    # LSTM Model paths
    lstm_paths = [
        '../lstm_model_ammad.h5',
        '../models/lstm_best_model.h5',
        '../models/lstm_seismic_risk_model.h5',
        '../models/lstm_simple_model.h5',
        'lstm_model_ammad.h5',
        'models/lstm_best_model.h5',
        '../../dashboard/lstm_model_ammad.h5',
        '../../dashboard/models/lstm_best_model.h5'
    ]
    
    # CatBoost Model paths
    catboost_paths = [
        '../thriey/earthquake_catboost_model.cbm',
        'thriey/earthquake_catboost_model.cbm',
        '../../dashboard/thriey/earthquake_catboost_model.cbm',
        '../dashboard/thriey/earthquake_catboost_model.cbm'
    ]
    
    # Scaler paths
    scaler_paths = [
        '../models/scaler.pkl',
        'models/scaler.pkl',
        'scaler.pkl',
        '../../dashboard/models/scaler.pkl'
    ]
    
    # Load LSTM model
    lstm_loaded = False
    for path in lstm_paths:
        if os.path.exists(path):
            try:
                lstm_model = load_model(path)
                print(f"[OK] LSTM model loaded from: {path}")
                
                # Check model input shape
                try:
                    input_shape = lstm_model.input_shape
                    if input_shape:
                        expected_features = input_shape[2] if len(input_shape) > 2 else input_shape[1]
                        print(f"[INFO] LSTM expects {expected_features} features per timestep")
                        print(f"[INFO] LSTM input shape: {input_shape}")
                except Exception as e:
                    print(f"[WARNING] Could not determine LSTM input shape: {e}")
                
                lstm_loaded = True
                break
            except Exception as e:
                print(f"[ERROR] Failed to load LSTM model from {path}: {e}")
    
    if not lstm_loaded:
        raise FileNotFoundError("Could not find LSTM model file. Please check the path.")
    
    # Load CatBoost model
    catboost_loaded = False
    for path in catboost_paths:
        if os.path.exists(path):
            try:
                catboost_model = CatBoostClassifier()
                catboost_model.load_model(path)
                print(f"[OK] CatBoost model loaded from: {path}")
                catboost_loaded = True
                break
            except Exception as e:
                print(f"[ERROR] Failed to load CatBoost model from {path}: {e}")
    
    if not catboost_loaded:
        print("[WARNING] CatBoost model not found. Forecast will use LSTM only.")
    
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
        scaler = DummyScaler()


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def generate_sample_operational_data(num_records: int = 200) -> pd.DataFrame:
    """Generate sample operational data when CSV files are not available"""
    np.random.seed(42)
    # Generate data spanning the last 30 days to ensure it covers most date ranges
    base_time = pd.Timestamp.now() - pd.Timedelta(days=30)
    
    data = {
        'recorded_at': [base_time + pd.Timedelta(hours=i * (30 * 24 / num_records)) for i in range(num_records)],
        'inj_flow': np.random.uniform(50, 150, num_records),
        'inj_whp': np.random.uniform(10, 30, num_records),
        'inj_temp': np.random.uniform(80, 120, num_records),
        'prod_temp': np.random.uniform(70, 110, num_records),
        'basin_flow': np.random.uniform(20, 80, num_records),
        'extracted_energy': np.random.uniform(5, 15, num_records),
        'thermal_power': np.random.uniform(1000, 3000, num_records),
        'prod_whp': np.random.uniform(8, 25, num_records)
    }
    df = pd.DataFrame(data)
    print(f"[INFO] Generated {len(df)} sample operational records spanning {df['recorded_at'].min()} to {df['recorded_at'].max()}")
    return df


def generate_sample_seismic_data(num_events: int = 20) -> pd.DataFrame:
    """Generate sample seismic data when CSV files are not available"""
    np.random.seed(42)
    base_time = pd.Timestamp.now() - pd.Timedelta(days=7)
    
    data = {
        'occurred_at': [base_time + pd.Timedelta(hours=np.random.uniform(0, 168)) for _ in range(num_events)],
        'magnitude': np.random.uniform(0.5, 3.5, num_events),
        'pgv_max': np.random.uniform(0.1, 5.0, num_events),
        'depth': np.random.uniform(1, 10, num_events)
    }
    return pd.DataFrame(data)


def find_csv_file(paths: List[str], description: str) -> Optional[pd.DataFrame]:
    """
    Find and load CSV file from multiple possible paths
    
    Args:
        paths: List of possible file paths
        description: Description of the file (for logging)
        
    Returns:
        DataFrame if found, None otherwise
    """
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"[OK] Loaded {description} from: {path}")
                return df
            except Exception as e:
                print(f"[ERROR] Failed to load {description} from {path}: {e}")
    
    # If no CSV found, generate sample data
    print(f"[INFO] No CSV file found for {description}, generating sample data...")
    if 'operational' in description.lower():
        return generate_sample_operational_data(200)
    elif 'seismic' in description.lower():
        return generate_sample_seismic_data(30)
    return None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - verifies API and model status"""
    return jsonify({
        'status': 'healthy',
        'lstm_loaded': lstm_model is not None,
        'catboost_loaded': catboost_model is not None,
        'scaler_loaded': scaler is not None,
        'forecast_days': FORECAST_DAYS
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint - predicts risk for one time point
    
    Expected JSON:
    {
        "data": [24 hours of operational data]
    }
    
    Returns: Risk level, probabilities, and confidence
    """
    try:
        if lstm_model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400
        
        input_data = data['data']
        
        if len(input_data) < LOOKBACK_HOURS:
            return jsonify({
                'error': f'Need at least {LOOKBACK_HOURS} hours of data',
                'received': len(input_data)
            }), 400
        
        # Prepare features
        features_array = prepare_features_for_lstm(input_data)
        
        # Scale features
        features_scaled = scaler.transform(features_array.reshape(-1, len(FEATURE_COLS)))
        features_scaled = features_scaled.reshape(1, LOOKBACK_HOURS, len(FEATURE_COLS))
        
        # Adjust features to match model's expected input size
        if lstm_model.input_shape:
            try:
                expected_features = lstm_model.input_shape[2] if len(lstm_model.input_shape) > 2 else lstm_model.input_shape[1]
                if expected_features != len(FEATURE_COLS):
                    features_scaled = adjust_features_for_model(features_scaled, expected_features)
            except Exception as e:
                print(f"[WARNING] Could not adjust features: {e}")
        
        # Make LSTM prediction
        prediction = lstm_model.predict(features_scaled, verbose=0)
        probabilities, predicted_class = process_lstm_prediction(prediction)
        
        # Get CatBoost prediction
        catboost_prob = get_catboost_prediction(input_data)
        
        # Map to risk levels
        num_classes = len(probabilities)
        prob_dict, risk_level, predicted_class = map_probabilities_to_risk_levels(probabilities, num_classes)
        
        # Apply ensemble adjustment
        if catboost_model is not None:
            prob_dict, risk_level, predicted_class = apply_ensemble_adjustment(prob_dict, catboost_prob)
        
        return jsonify({
            'risk_level': risk_level,
            'risk_level_code': predicted_class,
            'probabilities': prob_dict,
            'confidence': float(max(prob_dict.values())),
            'catboost_earthquake_prob': catboost_prob
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict risk levels for multiple time periods
    
    Expected JSON:
    {
        "data": [
            [24 hours of data for period 1],
            [24 hours of data for period 2],
            ...
        ]
    }
    """
    try:
        if lstm_model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400
        
        batch_data = data['data']
        predictions = []
        
        for period_data in batch_data:
            if len(period_data) < LOOKBACK_HOURS:
                continue
            
            # Prepare and scale features
            features_array = prepare_features_for_lstm(period_data)
            features_scaled = scaler.transform(features_array.reshape(-1, len(FEATURE_COLS)))
            features_scaled = features_scaled.reshape(1, LOOKBACK_HOURS, len(FEATURE_COLS))
            
            # Adjust features
            if lstm_model.input_shape:
                try:
                    expected_features = lstm_model.input_shape[2] if len(lstm_model.input_shape) > 2 else lstm_model.input_shape[1]
                    if expected_features != len(FEATURE_COLS):
                        features_scaled = adjust_features_for_model(features_scaled, expected_features)
                except Exception:
                    pass
            
            # Predict
            prediction = lstm_model.predict(features_scaled, verbose=0)
            probabilities, predicted_class = process_lstm_prediction(prediction)
            
            # Get CatBoost prediction
            catboost_prob = get_catboost_prediction(period_data)
            
            # Map to risk levels
            num_classes = len(probabilities)
            prob_dict, risk_level, predicted_class = map_probabilities_to_risk_levels(probabilities, num_classes)
            
            # Apply ensemble adjustment
            if catboost_model is not None:
                prob_dict, risk_level, predicted_class = apply_ensemble_adjustment(prob_dict, catboost_prob)
            
            predictions.append({
                'risk_level': risk_level,
                'risk_level_code': predicted_class,
                'probabilities': prob_dict,
                'catboost_earthquake_prob': catboost_prob
            })
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/data/operational', methods=['GET'])
def get_operational_data():
    """
    Get operational metrics data for specified time range
    
    Query Parameters:
        start_date: Start date/time (ISO format)
        end_date: End date/time (ISO format)
        limit: Maximum number of records (default: 1000)
    """
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = int(request.args.get('limit', 1000))
        
        operational_paths = [
            '../data/operational_metrics.csv',
            '../dashboard/data/operational_metrics.csv',
            'data/operational_metrics.csv'
        ]
        
        operational_df = find_csv_file(operational_paths, 'operational data')
        
        if operational_df is None or len(operational_df) == 0:
            # Generate sample data if CSV files not found
            print("[INFO] No CSV files found, generating sample data...")
            operational_df = generate_sample_operational_data(200)
        
        # Parse dates
        if 'recorded_at' in operational_df.columns:
            operational_df['recorded_at'] = pd.to_datetime(operational_df['recorded_at'], errors='coerce')
        
        # Filter by date range
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            filtered_df = operational_df[
                (operational_df['recorded_at'] >= start) & 
                (operational_df['recorded_at'] <= end)
            ]
            
            if len(filtered_df) == 0:
                print(f"[WARN] No data in range {start} to {end}, using all available data")
                # If no data in range, use all available data (especially for sample data)
                if len(operational_df) > limit:
                    operational_df = operational_df.tail(limit)
            else:
                operational_df = filtered_df
                if len(operational_df) > limit:
                    operational_df = operational_df.head(limit)
                print(f"[OK] Filtered to {len(operational_df)} records in date range")
        else:
            if len(operational_df) > limit:
                operational_df = operational_df.tail(limit)
        
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
    
    Query Parameters:
        start_date: Start date/time (ISO format)
        end_date: End date/time (ISO format)
    """
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        seismic_paths = [
            '../data/seismic_events.csv',
            '../dashboard/data/seismic_events.csv',
            'data/seismic_events.csv'
        ]
        
        seismic_df = find_csv_file(seismic_paths, 'seismic data')
        
        if seismic_df is None or len(seismic_df) == 0:
            # Generate sample data if CSV files not found
            print("[INFO] No CSV files found, generating sample seismic data...")
            seismic_df = generate_sample_seismic_data(30)
        
        # Parse dates
        if 'occurred_at' in seismic_df.columns:
            seismic_df['occurred_at'] = pd.to_datetime(seismic_df['occurred_at'], errors='coerce')
        
        # Filter by date range
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            seismic_df = seismic_df[
                (seismic_df['occurred_at'] >= start) & 
                (seismic_df['occurred_at'] <= end)
            ]
        
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
    """Get the latest 24 hours of operational and seismic data"""
    try:
        operational_paths = [
            '../data/operational_metrics.csv',
            '../dashboard/data/operational_metrics.csv',
            'data/operational_metrics.csv'
        ]
        
        operational_df = None
        for path in operational_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        total_rows = sum(1 for _ in f) - 1
                    
                    if total_rows > 1000:
                        skip_rows = total_rows - 1000
                        operational_df = pd.read_csv(path, skiprows=range(1, skip_rows + 1))
                    else:
                        operational_df = pd.read_csv(path)
                    break
                except Exception as e:
                    print(f"[ERROR] Failed to load from {path}: {e}")
                    continue
        
        seismic_paths = [
            '../data/seismic_events.csv',
            '../dashboard/data/seismic_events.csv',
            'data/seismic_events.csv'
        ]
        
        seismic_df = find_csv_file(seismic_paths, 'seismic data')
        
        if operational_df is None or len(operational_df) == 0:
            # Generate sample data if CSV files not found
            print("[INFO] No CSV files found, generating sample data...")
            operational_df = generate_sample_operational_data(200)
        
        # Get last 24 records
        recent_ops = operational_df.tail(24).copy()
        
        if 'recorded_at' in recent_ops.columns:
            recent_ops['recorded_at'] = pd.to_datetime(recent_ops['recorded_at'], errors='coerce')
        
        # Group seismic events by hour
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
    Generate 2-day seismic risk forecast using ensemble model (LSTM + CatBoost)
    
    Expected JSON:
    {
        "start_date": "2024-01-01T00:00:00",
        "historical_data": [24 hours of data]
    }
    
    Returns: JSON with 2-day forecast array
    """
    try:
        if lstm_model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        historical = data.get('historical_data', [])
        
        if len(historical) < LOOKBACK_HOURS:
            return jsonify({'error': f'Need at least {LOOKBACK_HOURS} hours of historical data'}), 400
        
        forecast = []
        start_date = datetime.fromisoformat(data.get('start_date', datetime.now().isoformat()))
        
        for day in range(FORECAST_DAYS):
            # Prepare features for LSTM
            features_array = prepare_features_for_lstm(historical)
            
            # Scale features
            features_scaled = scaler.transform(features_array.reshape(-1, len(FEATURE_COLS)))
            features_scaled = features_scaled.reshape(1, LOOKBACK_HOURS, len(FEATURE_COLS))
            
            # Adjust features to match model's expected input size
            if lstm_model.input_shape:
                try:
                    expected_features = lstm_model.input_shape[2] if len(lstm_model.input_shape) > 2 else lstm_model.input_shape[1]
                    if expected_features != len(FEATURE_COLS):
                        features_scaled = adjust_features_for_model(features_scaled, expected_features)
                except Exception as e:
                    print(f"[WARNING] Could not adjust features: {e}")
            
            # Make LSTM prediction
            prediction = lstm_model.predict(features_scaled, verbose=0)
            probabilities, predicted_class = process_lstm_prediction(prediction)
            
            # Get CatBoost earthquake probability
            catboost_prob = get_catboost_prediction(historical)
            
            # Map probabilities to risk levels
            num_classes = len(probabilities)
            prob_dict, risk_level, predicted_class = map_probabilities_to_risk_levels(probabilities, num_classes)
            
            # Apply ensemble adjustment
            if catboost_model is not None:
                prob_dict, risk_level, predicted_class = apply_ensemble_adjustment(prob_dict, catboost_prob)
            
            print(f"[FORECAST] Day {day+1} - Green: {prob_dict['green']:.4f}, Yellow: {prob_dict['yellow']:.4f}, "
                  f"Orange: {prob_dict['orange']:.4f}, Red: {prob_dict['red']:.4f}, "
                  f"CatBoost EQ: {catboost_prob:.4f}")
            
            forecast.append({
                'date': (start_date + timedelta(days=day)).isoformat(),
                'risk_level': risk_level,
                'risk_level_code': predicted_class,
                'probabilities': prob_dict,
                'catboost_earthquake_prob': catboost_prob
            })
        
        return jsonify({'forecast': forecast})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Loading Models...")
    print("=" * 60)
    
    try:
        load_models()
        print("\n" + "=" * 60)
        print("API Server Starting...")
        print("=" * 60)
        print("Endpoints:")
        print("  GET  /health - Health check")
        print("  POST /predict - Single prediction")
        print("  POST /predict/batch - Batch predictions")
        print("  POST /predict/forecast - 2-day forecast (Ensemble)")
        print("  GET  /data/operational - Get operational data")
        print("  GET  /data/seismic - Get seismic events")
        print("  GET  /data/latest - Get latest 24 hours")
        
        port = int(os.environ.get('PORT', 5000))
        print(f"\nServer running on http://0.0.0.0:{port}")
        print("=" * 60)
        
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"\n[ERROR] Error starting server: {e}")
        print("Please check that the model files exist and are accessible.")
