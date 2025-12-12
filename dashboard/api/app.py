"""
============================================================================
FLASK API FOR SEISMIC RISK PREDICTION - CATBOOST MODELS
============================================================================
Backend API server using 3 CatBoost models for seismic risk prediction
Adapted from latest_train_earth.py Dash functionality
============================================================================
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (relative to api folder)
BASE_PATH = Path(__file__).parent.parent.parent / 'latest'
model_event_path = BASE_PATH / "seismic_event_occurrence_model_v2.cbm"
model_magnitude_path = BASE_PATH / "seismic_magnitude_model_v2.cbm"
model_traffic_path = BASE_PATH / "seismic_traffic_light_3class_model_v2.cbm"
medians_path = BASE_PATH / "train_medians_v2.pkl"
threshold_path = BASE_PATH / "optimal_event_threshold_v2.txt"
data_path = BASE_PATH / "operational_seismic_linear_decay121.csv"

# Traffic light mapping
TRAFFIC_LABELS = {0: 'GREEN', 1: 'YELLOW', 2: 'RED'}
TRAFFIC_COLORS = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'}

# Operational variables with labels
OPERATIONAL_VARS = {
    'inj_flow': 'Injection Flow (m³/h)',
    'inj_whp': 'Injection Pressure (bar)',
    'inj_temp': 'Injection Temperature (°C)',
    'inj_ap': 'Injection Annular Pressure (bar)',
    'prod_temp': 'Production Temperature (°C)',
    'prod_whp': 'Production Pressure (bar)',
    'gt03_whp': 'GT03 Wellhead Pressure (bar)',
    'hedh_thpwr': 'Thermal Power (kW)',
    'basin_flow': 'Basin Flow (m³/h)',
    'prod_flow': 'Production Flow (m³/h)',
    'volume': 'Injected Volume (m³)',
    'cum_volume': 'Cumulative Volume (m³)',
    'inj_energy': 'Injected Energy (MWh)',
    'cum_inj_energy': 'Cumulative Energy (MWh)',
    'cooling_energy': 'Cooling Energy (MWh)',
    'cum_cooling_energy': 'Cumulative Cooling Energy (MWh)',
    'heat_exch_energy': 'Heat Exchanger Energy (MWh)',
}

# ============================================================================
# INITIALIZE FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)

# Global variables
models_loaded = False
df_dashboard = None
has_ground_truth = False
min_date = None
max_date = None

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Root endpoint - API information"""
    return jsonify({
        'service': 'Seismic Risk Prediction API',
        'status': 'running',
        'version': '2.0',
        'models_loaded': models_loaded,
        'endpoints': {
            'health': '/health',
            'info': '/data/info',
            'operational': '/data/operational',
            'events': '/data/events',
            'statistics': '/statistics'
        },
        'documentation': 'https://github.com/kiflomhailu/project_datascience_secondYear'
    })

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

def load_models_and_data():
    """Load CatBoost models and prepare dashboard data"""
    global models_loaded, df_dashboard, has_ground_truth, min_date, max_date
    
    print("\n" + "="*80)
    print("🚀 Loading models and data...")
    print("="*80)
    
    # Load models with error handling
    print("Loading models...")
    try:
        model_event = CatBoostClassifier()
        model_event.load_model(str(model_event_path))
        print("✓ Event occurrence model loaded")
        
        model_magnitude = CatBoostRegressor()
        model_magnitude.load_model(str(model_magnitude_path))
        print("✓ Magnitude model loaded")
        
        model_traffic = CatBoostClassifier()
        model_traffic.load_model(str(model_traffic_path))
        print("✓ Traffic light model loaded")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model files - {e}")
        print("API will continue without prediction capabilities")
        return False
    
    # Load training medians
    with open(medians_path, 'rb') as f:
        train_medians = pickle.load(f)
    print("✓ Training medians loaded")
    
    # Load optimal threshold
    with open(threshold_path, 'r') as f:
        optimal_threshold = float(f.read().strip())
    print(f"✓ Optimal threshold loaded: {optimal_threshold:.6f}")
    
    # Load operational data
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✓ Loaded {len(df):,} records")
    
    # ==========================================
    # PREPROCESSING
    # ==========================================
    
    # Replace sentinel values
    sentinel_cols = ['pgv_max', 'magnitude', 'hourly_seismicity_rate']
    for col in sentinel_cols:
        if col in df.columns:
            mask = df[col] == -999.0
            df.loc[mask, col] = 0
    
    # Parse datetime
    datetime_cols = ['recorded_at', 'phase_started_at', 'phase_production_ended_at',
                     'phase_ended_at', 'occurred_at']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Sort chronologically
    if 'recorded_at' in df.columns:
        df = df.sort_values('recorded_at').reset_index(drop=True)
        min_date = df['recorded_at'].min()
        max_date = df['recorded_at'].max()
    else:
        df = df.reset_index(drop=True)
    
    # Create ground truth targets
    if 'magnitude' in df.columns and 'hourly_seismicity_rate' in df.columns:
        df['event_occurs'] = ((df['magnitude'] >= 0.17) | (df['hourly_seismicity_rate'] > 0)).astype(int)
        
        def classify_traffic_light_3class(magnitude):
            if magnitude >= 1.0:
                return 2  # RED
            elif magnitude >= 0.17:
                return 1  # YELLOW
            else:
                return 0  # GREEN
        
        df['traffic_light_actual'] = df['magnitude'].apply(classify_traffic_light_3class)
        df['magnitude_actual'] = df['magnitude'].copy()
        has_ground_truth = True
    
    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================
    
    # Temporal features
    if 'recorded_at' in df.columns:
        df['hour'] = df['recorded_at'].dt.hour
        df['day_of_week'] = df['recorded_at'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['recorded_at'].dt.month
    
    # Operational phase duration
    if 'phase_started_at' in df.columns and 'recorded_at' in df.columns:
        df['phase_duration_hours'] = (df['recorded_at'] - df['phase_started_at']).dt.total_seconds() / 3600
    
    # Rolling statistics
    for window in [6, 12, 24]:
        if 'inj_temp' in df.columns:
            df[f'inj_temp_rolling_mean_{window}h'] = df['inj_temp'].rolling(window, min_periods=1).mean()
            df[f'inj_temp_rolling_std_{window}h'] = df['inj_temp'].rolling(window, min_periods=1).std()
        if 'inj_whp' in df.columns:
            df[f'inj_whp_rolling_mean_{window}h'] = df['inj_whp'].rolling(window, min_periods=1).mean()
        if 'prod_flow' in df.columns:
            df[f'prod_flow_rolling_max_{window}h'] = df['prod_flow'].rolling(window, min_periods=1).max()
    
    # Rate of change
    for c in ['inj_temp', 'inj_whp', 'cum_inj_energy', 'prod_temp']:
        if c in df.columns:
            df[f'{c}_change'] = df[c].diff()
    
    # Pressure and temperature differences
    if 'inj_whp' in df.columns and 'prod_whp' in df.columns:
        df['pressure_diff'] = df['inj_whp'] - df['prod_whp']
    if 'inj_temp' in df.columns and 'prod_temp' in df.columns:
        df['temp_diff'] = df['inj_temp'] - df['prod_temp']
    
    # Energy efficiency metrics
    if 'inj_energy' in df.columns and 'inj_flow' in df.columns:
        df['inj_energy_per_flow'] = df['inj_energy'] / (df['inj_flow'] + 1e-6)
    if 'cooling_energy' in df.columns and 'inj_energy' in df.columns:
        df['cooling_efficiency'] = df['cooling_energy'] / (df['inj_energy'] + 1e-6)
    
    # Cumulative stress indicators
    if 'cum_inj_energy' in df.columns and 'cum_volume' in df.columns:
        df['cum_energy_normalized'] = df['cum_inj_energy'] / (df['cum_volume'] + 1e-6)
    
    # Interaction features
    if 'inj_temp' in df.columns and 'inj_whp' in df.columns:
        df['temp_pressure_interaction'] = df['inj_temp'] * df['inj_whp']
    if 'inj_flow' in df.columns and 'inj_whp' in df.columns:
        df['flow_pressure_interaction'] = df['inj_flow'] * df['inj_whp']
    
    print("✓ Feature engineering complete")
    
    # ==========================================
    # PREPARE FEATURES
    # ==========================================
    
    exclude_cols = [
        'recorded_at', 'phase_started_at', 'phase_production_ended_at',
        'phase_ended_at', 'occurred_at', 'event_occurs', 'event_magnitude',
        'traffic_light', 'traffic_light_actual', 'magnitude', 'magnitude_actual',
        'hourly_seismicity_rate', 'rounded', 'adjusted'
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_operational = df[feature_cols].copy()
    
    # ==========================================
    # IMPUTATION
    # ==========================================
    
    numeric_cols = X_operational.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_operational.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Replace infinities
    X_operational = X_operational.replace([np.inf, -np.inf], np.nan)
    
    # Fill numeric NaNs with training medians
    X_operational.loc[:, numeric_cols] = X_operational[numeric_cols].fillna(train_medians)
    
    # Fill categorical NaNs
    for col in categorical_cols:
        X_operational.loc[:, col] = X_operational[col].astype(str).fillna('missing')
    
    print(f"✓ Data preprocessing complete: {X_operational.shape[1]} features")
    
    # ==========================================
    # MAKE PREDICTIONS
    # ==========================================
    
    cat_features = [i for i, col in enumerate(X_operational.columns) if col in categorical_cols]
    operational_pool = Pool(X_operational, cat_features=cat_features)
    
    print("\nMaking predictions...")
    
    # Event occurrence probability
    y_event_prob = model_event.predict_proba(operational_pool)[:, 1]
    y_event_pred = (y_event_prob >= optimal_threshold).astype(int)
    print(f"✓ Event predictions complete ({y_event_pred.sum():,} events predicted)")
    
    # Magnitude prediction
    y_magnitude_pred = np.zeros(len(X_operational))
    if y_event_pred.sum() > 0:
        X_predicted_events = X_operational.iloc[y_event_pred == 1].reset_index(drop=True)
        event_pool = Pool(X_predicted_events, cat_features=cat_features)
        y_magnitude_pred[y_event_pred == 1] = model_magnitude.predict(event_pool)
        print(f"✓ Magnitude predictions complete")
    
    # Traffic light prediction
    y_traffic_pred = model_traffic.predict(operational_pool).flatten()
    print(f"✓ Traffic light predictions complete")
    
    # ==========================================
    # PREPARE DASHBOARD DATA
    # ==========================================
    
    df_dashboard = pd.DataFrame({
        'event_probability': y_event_prob,
        'event_predicted': y_event_pred,
        'magnitude_predicted': y_magnitude_pred,
        'traffic_light_pred': y_traffic_pred
    })
    
    # Add timestamp
    if 'recorded_at' in df.columns:
        df_dashboard['timestamp'] = df['recorded_at'].values
    
    # Add operational variables
    for col in OPERATIONAL_VARS.keys():
        if col in df.columns:
            df_dashboard[col] = df[col].values
    
    # Add ground truth
    if has_ground_truth:
        df_dashboard['event_actual'] = df['event_occurs'].values
        df_dashboard['magnitude_actual'] = df['magnitude_actual'].values
        df_dashboard['traffic_light_actual'] = df['traffic_light_actual'].values
    
    # Map traffic light to labels and colors
    df_dashboard['traffic_label'] = df_dashboard['traffic_light_pred'].map(TRAFFIC_LABELS)
    df_dashboard['traffic_color'] = df_dashboard['traffic_light_pred'].map(TRAFFIC_COLORS)
    
    print(f"\n✓ Dashboard data prepared: {len(df_dashboard):,} samples")
    print("="*80)
    
    models_loaded = True
    return optimal_threshold

# Load everything at startup
try:
    optimal_threshold = load_models_and_data()
except Exception as e:
    print(f"❌ Error loading models: {e}")
    optimal_threshold = 0.5

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok' if models_loaded else 'error',
        'models_loaded': models_loaded,
        'records': len(df_dashboard) if df_dashboard is not None else 0,
        'has_ground_truth': has_ground_truth
    })

@app.route('/data/info', methods=['GET'])
def get_data_info():
    """Get dataset information"""
    if df_dashboard is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    return jsonify({
        'total_records': len(df_dashboard),
        'min_date': min_date.isoformat() if min_date else None,
        'max_date': max_date.isoformat() if max_date else None,
        'available_variables': list(OPERATIONAL_VARS.keys()),
        'variable_labels': OPERATIONAL_VARS,
        'has_ground_truth': has_ground_truth
    })

@app.route('/data/operational', methods=['GET'])
def get_operational_data():
    """Get operational data with optional date filtering"""
    if df_dashboard is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    variables = request.args.get('variables')  # Comma-separated
    
    df_filtered = df_dashboard.copy()
    
    # Date filtering
    if start_date and 'timestamp' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['timestamp'] >= pd.to_datetime(start_date)]
    if end_date and 'timestamp' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['timestamp'] <= pd.to_datetime(end_date)]
    
    # Convert timestamps to ISO format
    if 'timestamp' in df_filtered.columns:
        df_filtered['timestamp'] = df_filtered['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Variable selection
    if variables:
        var_list = variables.split(',')
        base_cols = ['timestamp', 'event_probability', 'event_predicted', 
                     'magnitude_predicted', 'traffic_light_pred', 'traffic_label', 'traffic_color']
        if has_ground_truth:
            base_cols.extend(['event_actual', 'magnitude_actual', 'traffic_light_actual'])
        
        selected_cols = base_cols + [v for v in var_list if v in df_filtered.columns]
        df_filtered = df_filtered[selected_cols]
    
    # Sample for performance if too large
    if len(df_filtered) > 10000:
        df_filtered = df_filtered.iloc[::max(1, len(df_filtered)//10000)]
    
    return jsonify({
        'data': df_filtered.to_dict('records'),
        'count': len(df_filtered)
    })

@app.route('/data/events', methods=['GET'])
def get_events():
    """Get detected seismic events"""
    if df_dashboard is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    threshold = float(request.args.get('threshold', optimal_threshold))
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df_filtered = df_dashboard.copy()
    
    # Date filtering
    if start_date and 'timestamp' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['timestamp'] >= pd.to_datetime(start_date)]
    if end_date and 'timestamp' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['timestamp'] <= pd.to_datetime(end_date)]
    
    # Apply threshold
    df_filtered['event_pred_dynamic'] = (df_filtered['event_probability'] >= threshold).astype(int)
    events_df = df_filtered[df_filtered['event_pred_dynamic'] == 1].copy()
    
    # Add traffic light label
    events_df['traffic_light'] = events_df['traffic_label']
    
    if 'timestamp' in events_df.columns:
        events_df['timestamp'] = events_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return jsonify({
        'events': events_df.to_dict('records'),
        'count': len(events_df),
        'threshold': threshold
    })

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get overall statistics"""
    if df_dashboard is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    threshold = float(request.args.get('threshold', optimal_threshold))
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df_filtered = df_dashboard.copy()
    
    # Date filtering
    if start_date and 'timestamp' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['timestamp'] >= pd.to_datetime(start_date)]
    if end_date and 'timestamp' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['timestamp'] <= pd.to_datetime(end_date)]
    
    # Apply threshold
    df_filtered['event_pred_dynamic'] = (df_filtered['event_probability'] >= threshold).astype(int)
    
    n_events = df_filtered['event_pred_dynamic'].sum()
    pct_events = (n_events / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    
    green_count = (df_filtered['traffic_light_pred'] == 0).sum()
    yellow_count = (df_filtered['traffic_light_pred'] == 1).sum()
    red_count = (df_filtered['traffic_light_pred'] == 2).sum()
    
    events_df = df_filtered[df_filtered['event_pred_dynamic'] == 1]
    if n_events > 0:
        avg_magnitude = float(events_df['magnitude_predicted'].mean())
        max_magnitude = float(events_df['magnitude_predicted'].max())
    else:
        avg_magnitude = 0.0
        max_magnitude = 0.0
    
    stats = {
        'total_samples': int(len(df_filtered)),
        'events_detected': int(n_events),
        'events_percentage': float(pct_events),
        'green_count': int(green_count),
        'yellow_count': int(yellow_count),
        'red_count': int(red_count),
        'avg_magnitude': avg_magnitude,
        'max_magnitude': max_magnitude,
        'threshold': threshold
    }
    
    # Add confusion matrix if ground truth available
    if has_ground_truth:
        from sklearn.metrics import confusion_matrix
        y_true = df_filtered['event_actual'].values
        y_pred = df_filtered['event_pred_dynamic'].values
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        accuracy = float((tp + tn) / (tp + tn + fp + fn))
        f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        stats['confusion_matrix'] = {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
        stats['metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    return jsonify(stats)

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting Seismic Risk Prediction API")
    print("="*80)
    print(f"📍 API URL: http://127.0.0.1:5000")
    print(f"\n📡 Available Endpoints:")
    print(f"   • GET  /health             - Health check")
    print(f"   • GET  /data/info          - Dataset information")
    print(f"   • GET  /data/operational   - Operational data (with filters)")
    print(f"   • GET  /data/events        - Detected events")
    print(f"   • GET  /statistics         - Statistics and metrics")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
