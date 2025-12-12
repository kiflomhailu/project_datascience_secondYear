

# ==========================================
# Seismic Traffic Light Dashboard v4 - Advanced Features
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from datetime import datetime, timedelta

# -----------------------------
# Paths
# -----------------------------
model_event_path = "seismic_event_occurrence_model_v2.cbm"
model_magnitude_path = "seismic_magnitude_model_v2.cbm"
model_traffic_path = "seismic_traffic_light_3class_model_v2.cbm"
medians_path = "train_medians_v2.pkl"
threshold_path = "optimal_event_threshold_v2.txt"
operational_data_path = "operational_seismic_linear_decay121.csv"

# -----------------------------
# Load trained models
# -----------------------------
print("Loading models...")
model_event = CatBoostClassifier()
model_event.load_model(model_event_path)
print("✓ Event occurrence model loaded")

model_magnitude = CatBoostRegressor()
model_magnitude.load_model(model_magnitude_path)
print("✓ Magnitude model loaded")

model_traffic = CatBoostClassifier()
model_traffic.load_model(model_traffic_path)
print("✓ Traffic light model loaded")

# Load training medians
with open(medians_path, 'rb') as f:
    train_medians = pickle.load(f)
print("✓ Training medians loaded")

# Load optimal threshold
with open(threshold_path, 'r') as f:
    optimal_threshold = float(f.read().strip())
print(f"✓ Optimal threshold loaded: {optimal_threshold:.6f}")

# -----------------------------
# Load operational data
# -----------------------------
df = pd.read_csv(operational_data_path, low_memory=False)
print(f"✓ Loaded {len(df):,} records")

# ==========================================
# PREPROCESSING (MUST MATCH TRAINING)
# ==========================================

# 1. Replace sentinel values (-999) with 0
sentinel_cols = ['pgv_max', 'magnitude', 'hourly_seismicity_rate']
for col in sentinel_cols:
    if col in df.columns:
        mask = df[col] == -999.0
        df.loc[mask, col] = 0

# 2. Parse datetime columns
datetime_cols = ['recorded_at', 'phase_started_at', 'phase_production_ended_at',
                 'phase_ended_at', 'occurred_at']
for col in datetime_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# 3. Sort chronologically
if 'recorded_at' in df.columns:
    df = df.sort_values('recorded_at').reset_index(drop=True)
    has_timestamp = True
    min_date = df['recorded_at'].min()
    max_date = df['recorded_at'].max()
else:
    df = df.reset_index(drop=True)
    has_timestamp = False

# Create ground truth targets if data exists
has_ground_truth = False
if 'magnitude' in df.columns and 'hourly_seismicity_rate' in df.columns:
    df['event_occurs'] = ((df['magnitude'] >= 0.17) | (df['hourly_seismicity_rate'] > 0)).astype(int)
    
    # 3-class traffic light
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
# FEATURE ENGINEERING (MUST MATCH TRAINING)
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
# IMPUTATION (SAME AS TRAINING)
# ==========================================

# Get numeric and categorical columns
numeric_cols = X_operational.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_operational.select_dtypes(exclude=[np.number]).columns.tolist()

# Replace infinities with NaN
X_operational = X_operational.replace([np.inf, -np.inf], np.nan)

# Fill numeric NaNs with training medians
X_operational.loc[:, numeric_cols] = X_operational[numeric_cols].fillna(train_medians)

# Fill categorical NaNs with 'missing'
for col in categorical_cols:
    X_operational.loc[:, col] = X_operational[col].astype(str).fillna('missing')

print(f"✓ Data preprocessing complete: {X_operational.shape[1]} features")

# ==========================================
# MAKE PREDICTIONS
# ==========================================

# Identify categorical feature indices
cat_features = [i for i, col in enumerate(X_operational.columns) if col in categorical_cols]

# Create Pool for predictions
operational_pool = Pool(X_operational, cat_features=cat_features)

print("\nMaking predictions...")

# Prediction 1: Event occurrence probability
y_event_prob = model_event.predict_proba(operational_pool)[:, 1]
y_event_pred = (y_event_prob >= optimal_threshold).astype(int)

print(f"✓ Event predictions complete ({y_event_pred.sum():,} events predicted)")

# Prediction 2: Magnitude (only for predicted events)
y_magnitude_pred = np.zeros(len(X_operational))
if y_event_pred.sum() > 0:
    X_predicted_events = X_operational.iloc[y_event_pred == 1].reset_index(drop=True)
    event_pool = Pool(X_predicted_events, cat_features=cat_features)
    y_magnitude_pred[y_event_pred == 1] = model_magnitude.predict(event_pool)
    print(f"✓ Magnitude predictions complete")

# Prediction 3: Traffic light (3-class)
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
if has_timestamp:
    df_dashboard['timestamp'] = df['recorded_at'].values

# Add ALL operational variables
operational_vars = {
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

for col, label in operational_vars.items():
    if col in df.columns:
        df_dashboard[col] = df[col].values

# Add ground truth if available
if has_ground_truth:
    df_dashboard['event_actual'] = df['event_occurs'].values
    df_dashboard['magnitude_actual'] = df['magnitude_actual'].values
    df_dashboard['traffic_light_actual'] = df['traffic_light_actual'].values

# Map traffic light to labels and colors
traffic_labels = {0: '🟢 GREEN', 1: '🟡 YELLOW', 2: '🔴 RED'}
traffic_colors = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'}

df_dashboard['traffic_label'] = df_dashboard['traffic_light_pred'].map(traffic_labels)
df_dashboard['traffic_color'] = df_dashboard['traffic_light_pred'].map(traffic_colors)

print(f"\n✓ Dashboard data prepared: {len(df_dashboard):,} samples")

# ==========================================
# DASH APP
# ==========================================

app = Dash(__name__)

if has_timestamp:
    # Available operational variables for checkboxes
    available_vars = [(col, label) for col, label in operational_vars.items() if col in df_dashboard.columns]
    
    app.layout = html.Div([
        html.H1("🚦 Advanced Seismic Monitoring Dashboard", 
                style={'textAlign':'center', 'color':'#2c3e50', 'marginBottom': 10}),
        html.H3("Real-time Operational Monitoring | Event Detection | Risk Assessment",
                style={'textAlign':'center', 'color':'#7f8c8d', 'marginBottom': 30}),
        
        # Controls Panel
        html.Div([
            # Date range
            html.Div([
                html.Label("Start Date:", style={'fontSize': 14, 'fontWeight': 'bold', 'marginRight': 10}),
                dcc.DatePickerSingle(
                    id='start_date',
                    date=min_date,
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    display_format='YYYY-MM-DD',
                ),
            ], style={'display': 'inline-block', 'marginRight': 30}),
            
            html.Div([
                html.Label("End Date:", style={'fontSize': 14, 'fontWeight': 'bold', 'marginRight': 10}),
                dcc.DatePickerSingle(
                    id='end_date',
                    date=min_date + timedelta(days=30),
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    display_format='YYYY-MM-DD',
                ),
            ], style={'display': 'inline-block', 'marginRight': 30}),
            
            # Threshold
            html.Div([
                html.Label("Event Threshold:", style={'fontSize': 14, 'fontWeight': 'bold', 'marginRight': 10}),
                dcc.Input(
                    id='threshold',
                    type='number',
                    value=optimal_threshold,
                    min=0.001,
                    max=0.99,
                    step=0.001,
                    style={'width': '100px', 'fontSize': 14, 'padding': '5px'}
                ),
            ], style={'display': 'inline-block', 'marginRight': 30}),
            
            # Show actual checkbox
            html.Div([
                dcc.Checklist(
                    id='show_actual',
                    options=[{'label': ' Show Actual Events', 'value': 'show'}],
                    value=['show'] if has_ground_truth else [],
                    style={'fontSize': 14, 'fontWeight': 'bold'}
                ),
            ], style={'display': 'inline-block'}) if has_ground_truth else html.Div(),
            
        ], style={'textAlign': 'center', 'marginBottom': 20, 'padding': '15px', 
                  'backgroundColor': '#ecf0f1', 'borderRadius': 10}),
        
        # Variable Selection Panel
        html.Div([
            html.H4("📊 Select Operational Variables to Display:", 
                   style={'marginBottom': 15, 'color': '#2c3e50'}),
            dcc.Checklist(
                id='variable_selector',
                options=[{'label': f' {label}', 'value': col} for col, label in available_vars],
                value=['inj_whp', 'prod_whp', 'inj_temp'] if len(available_vars) > 0 else [],
                inline=True,
                style={'fontSize': 13},
                labelStyle={'marginRight': '25px', 'marginBottom': '10px'}
            ),
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': 10, 'marginBottom': 20}),
        
        # Main plot
        dcc.Graph(id='main_plot', style={'height': '700px'}),
        
        # Statistics and Confusion Matrix Row
        html.Div([
            # Statistics
            html.Div(id='stats', style={
                'width': '58%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': 20,
                'backgroundColor': '#ecf0f1',
                'borderRadius': 10,
                'marginRight': '2%'
            }),
            
            # Confusion Matrix
            html.Div(id='confusion_matrix', style={
                'width': '38%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': 20,
                'backgroundColor': '#f8f9fa',
                'borderRadius': 10
            }) if has_ground_truth else html.Div(),
        ], style={'marginTop': 30}),
        
        # Event table with DataTable
        html.Div([
            html.H3("⚠️ Detected Events", style={'textAlign': 'center', 'color': '#e74c3c', 'marginBottom': 20}),
            dash_table.DataTable(
                id='event_table',
                page_size=15,
                filter_action='native',
                sort_action='native',
                sort_mode='multi',
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'fontSize': 13,
                    'fontFamily': 'Arial'
                },
                style_header={
                    'backgroundColor': '#34495e',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Risk Level} = "🔴 RED"'},
                        'backgroundColor': '#fadbd8',
                    },
                    {
                        'if': {'filter_query': '{Risk Level} = "🟡 YELLOW"'},
                        'backgroundColor': '#fcf3cf',
                    },
                    {
                        'if': {'filter_query': '{Risk Level} = "🟢 GREEN"'},
                        'backgroundColor': '#d5f4e6',
                    }
                ],
            ),
        ], style={'marginTop': 30, 'padding': 20, 'backgroundColor': '#ffffff', 'borderRadius': 10}),
        
    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'maxWidth': '1800px', 'margin': '0 auto'})

    @app.callback(
        [Output('main_plot', 'figure'),
         Output('stats', 'children'),
         Output('confusion_matrix', 'children') if has_ground_truth else Output('stats', 'children', allow_duplicate=True),
         Output('event_table', 'data'),
         Output('event_table', 'columns')],
        [Input('start_date', 'date'),
         Input('end_date', 'date'),
         Input('threshold', 'value'),
         Input('variable_selector', 'value'),
         Input('show_actual', 'value')],
        prevent_initial_call='initial_duplicate' if has_ground_truth else False
    )
    def update_dashboard(start_date, end_date, threshold, selected_vars, show_actual):
        # Filter data by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        mask = (df_dashboard['timestamp'] >= start_dt) & (df_dashboard['timestamp'] <= end_dt)
        df_filtered = df_dashboard[mask].copy()
        
        if len(df_filtered) == 0:
            # Empty responses
            fig = go.Figure()
            fig.add_annotation(text="No data in selected date range",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False,
                             font=dict(size=20))
            empty_stats = html.Div("No data available")
            empty_cm = html.Div()
            return fig, empty_stats, empty_cm, [], []
        
        # Update event predictions with new threshold
        df_filtered['event_pred_dynamic'] = (df_filtered['event_probability'] >= threshold).astype(int)
        
        # Determine number of y-axes needed
        n_vars = len(selected_vars) if selected_vars else 0
        
        # Create figure with multiple y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Color palette for variables
        var_colors = ['#3498db', '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#16a085', '#d35400']
        
        # Add selected operational variables
        for idx, var in enumerate(selected_vars) if selected_vars else []:
            if var in df_filtered.columns:
                var_label = operational_vars.get(var, var)
                fig.add_trace(
                    go.Scatter(
                        x=df_filtered['timestamp'],
                        y=df_filtered[var],
                        mode='lines',
                        name=var_label,
                        line=dict(color=var_colors[idx % len(var_colors)], width=1.5),
                        hovertemplate=f'{var_label}: %{{y:.2f}}<extra></extra>',
                        yaxis='y' if idx == 0 else f'y{idx+1}'
                    ),
                    secondary_y=False
                )
        
        # Add predicted events (on secondary y-axis for magnitude)
        events_df = df_filtered[df_filtered['event_pred_dynamic'] == 1].copy()
        
        if len(events_df) > 0:
            hover_text = []
            for idx, row in events_df.iterrows():
                text = f"<b>⚠️ SEISMIC EVENT</b><br>"
                text += f"Time: {row['timestamp']}<br>"
                text += f"Probability: {row['event_probability']:.4f}<br>"
                text += f"Magnitude: {row['magnitude_predicted']:.3f}<br>"
                text += f"Risk: {row['traffic_label']}<br>"
                for var in (selected_vars or []):
                    if var in row and pd.notna(row[var]):
                        var_label = operational_vars.get(var, var)
                        text += f"{var_label}: {row[var]:.2f}<br>"
                if has_ground_truth and 'magnitude_actual' in row:
                    text += f"Actual Mag: {row['magnitude_actual']:.3f}"
                hover_text.append(text)
            
            fig.add_trace(
                go.Scatter(
                    x=events_df['timestamp'],
                    y=events_df['magnitude_predicted'],
                    mode='markers',
                    name='Predicted Events',
                    marker=dict(
                        size=12,
                        color=events_df['traffic_color'],
                        line=dict(width=2, color='white'),
                        symbol='diamond'
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>'
                ),
                secondary_y=True
            )
        
        # Add actual events if checkbox is selected
        if has_ground_truth and show_actual and 'show' in show_actual:
            actual_events = df_filtered[df_filtered['event_actual'] == 1].copy()
            if len(actual_events) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=actual_events['timestamp'],
                        y=actual_events['magnitude_actual'],
                        mode='markers',
                        name='Actual Events',
                        marker=dict(size=10, color='black', symbol='x', line=dict(width=2)),
                        hovertemplate='Actual Magnitude: %{y:.3f}<extra></extra>'
                    ),
                    secondary_y=True
                )
        
        # Add magnitude threshold lines
        if len(events_df) > 0:
            fig.add_hline(y=0.17, line_dash="dash", line_color="orange", 
                         annotation_text="YELLOW", annotation_position="right",
                         secondary_y=True)
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                         annotation_text="RED", annotation_position="right",
                         secondary_y=True)
        
        # Update layout
        fig.update_xaxes(title_text="Time", showgrid=True, gridcolor='lightgray')
        
        if selected_vars and len(selected_vars) > 0:
            first_var_label = operational_vars.get(selected_vars[0], selected_vars[0])
            fig.update_yaxes(title_text=first_var_label, secondary_y=False, 
                            showgrid=True, gridcolor='lightgray')
        
        fig.update_yaxes(title_text="Event Magnitude", secondary_y=True, 
                        showgrid=False, range=[0, max(3, df_filtered['magnitude_predicted'].max() * 1.2)])
        
        fig.update_layout(
            title=dict(
                text=f"Seismic Monitoring: {start_date} to {end_date}",
                x=0.5,
                xanchor='center',
                font=dict(size=18, color='#2c3e50')
            ),
            hovermode='closest',
            plot_bgcolor='white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05,
                font=dict(size=11)
            ),
            height=700,
            margin=dict(r=150)
        )
        
        # Statistics
        n_events = df_filtered['event_pred_dynamic'].sum()
        pct_events = (n_events / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        
        green_count = (df_filtered['traffic_light_pred'] == 0).sum()
        yellow_count = (df_filtered['traffic_light_pred'] == 1).sum()
        red_count = (df_filtered['traffic_light_pred'] == 2).sum()
        
        if n_events > 0:
            avg_magnitude = events_df['magnitude_predicted'].mean()
            max_magnitude = events_df['magnitude_predicted'].max()
        else:
            avg_magnitude = 0
            max_magnitude = 0
        
        stats_text = html.Div([
            html.H4("📈 Statistics", style={'marginBottom': 15, 'color': '#2c3e50'}),
            html.Div([
                html.P(f"📊 Total Samples: {len(df_filtered):,}", style={'fontSize': 16, 'marginBottom': 10}),
                html.P(f"⚠️ Events Detected: {n_events:,} ({pct_events:.2f}%)", 
                      style={'fontSize': 16, 'marginBottom': 10, 'color': '#e74c3c', 'fontWeight': 'bold'}),
            ]),
            html.Div([
                html.Span(f"🟢 GREEN: {green_count:,}", 
                         style={'marginRight': 20, 'color': '#27ae60', 'fontWeight': 'bold', 'fontSize': 15}),
                html.Span(f"🟡 YELLOW: {yellow_count:,}", 
                         style={'marginRight': 20, 'color': '#f39c12', 'fontWeight': 'bold', 'fontSize': 15}),
                html.Span(f"🔴 RED: {red_count:,}", 
                         style={'color': '#e74c3c', 'fontWeight': 'bold', 'fontSize': 15}),
            ], style={'marginBottom': 15}),
            html.Div([
                html.P(f"📏 Avg Magnitude: {avg_magnitude:.3f}", style={'fontSize': 15, 'marginBottom': 5}),
                html.P(f"📈 Max Magnitude: {max_magnitude:.3f}", style={'fontSize': 15}),
            ]) if n_events > 0 else html.Div(html.P("✅ No events detected", style={'color': '#27ae60', 'fontSize': 16})),
        ])
        
        # Confusion Matrix
        cm_display = html.Div()
        if has_ground_truth:
            y_event_true = df_filtered['event_actual'].values
            y_event_pred = df_filtered['event_pred_dynamic'].values
            
            cm = confusion_matrix(y_event_true, y_event_pred)
            tn, fp, fn, tp = cm.ravel()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            cm_display = html.Div([
                html.H4("🎯 Confusion Matrix", style={'marginBottom': 15, 'color': '#2c3e50', 'textAlign': 'center'}),
                html.Div([
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("", style={'padding': 10, 'border': '1px solid #ddd'}),
                            html.Th("Pred: No", style={'padding': 10, 'border': '1px solid #ddd', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
                            html.Th("Pred: Event", style={'padding': 10, 'border': '1px solid #ddd', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td("Actual: No", style={'padding': 10, 'border': '1px solid #ddd', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}),
                                html.Td(f"{tn:,}", style={'padding': 10, 'border': '1px solid #ddd', 'backgroundColor': '#d5f4e6', 'textAlign': 'center', 'fontSize': 16, 'fontWeight': 'bold'}),
                                html.Td(f"{fp:,}", style={'padding': 10, 'border': '1px solid #ddd', 'backgroundColor': '#fadbd8', 'textAlign': 'center', 'fontSize': 16, 'fontWeight': 'bold'}),
                            ]),
                            html.Tr([
                                html.Td("Actual: Event", style={'padding': 10, 'border': '1px solid #ddd', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}),
                                html.Td(f"{fn:,}", style={'padding': 10, 'border': '1px solid #ddd', 'backgroundColor': '#fadbd8', 'textAlign': 'center', 'fontSize': 16, 'fontWeight': 'bold'}),
                                html.Td(f"{tp:,}", style={'padding': 10, 'border': '1px solid #ddd', 'backgroundColor': '#d5f4e6', 'textAlign': 'center', 'fontSize': 16, 'fontWeight': 'bold'}),
                            ]),
                        ])
                    ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': 20}),
                ], style={'marginBottom': 15}),
                html.Div([
                    html.Div([
                        html.Span("Accuracy: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{accuracy*100:.2f}%", style={'color': '#27ae60' if accuracy > 0.9 else '#e67e22'})
                    ], style={'fontSize': 15, 'marginBottom': 8}),
                    html.Div([
                        html.Span("Precision: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{precision*100:.2f}%", style={'color': '#27ae60' if precision > 0.7 else '#e67e22'})
                    ], style={'fontSize': 15, 'marginBottom': 8}),
                    html.Div([
                        html.Span("Recall: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{recall*100:.2f}%", style={'color': '#27ae60' if recall > 0.7 else '#e67e22'})
                    ], style={'fontSize': 15, 'marginBottom': 8}),
                    html.Div([
                        html.Span("F1-Score: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{f1*100:.2f}%", style={'color': '#27ae60' if f1 > 0.7 else '#e67e22'})
                    ], style={'fontSize': 15}),
                ], style={'textAlign': 'center'})
            ])
        
        # Event Table Data
        table_data = []
        table_columns = []
        
        if n_events > 0:
            # Prepare table data
            events_display = events_df.copy()
            events_display['Time'] = events_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            events_display['Probability'] = events_display['event_probability'].round(4)
            events_display['Magnitude'] = events_display['magnitude_predicted'].round(3)
            events_display['Risk Level'] = events_display['traffic_label']
            
            # Add selected operational variables to table
            table_cols = ['Time', 'Probability', 'Magnitude', 'Risk Level']
            for var in (selected_vars or []):
                if var in events_display.columns:
                    var_label = operational_vars.get(var, var)
                    events_display[var_label] = events_display[var].round(2)
                    table_cols.append(var_label)
            
            # Add actual magnitude if available
            if has_ground_truth and 'magnitude_actual' in events_display.columns:
                events_display['Actual Magnitude'] = events_display['magnitude_actual'].round(3)
                table_cols.append('Actual Magnitude')
            
            table_data = events_display[table_cols].to_dict('records')
            table_columns = [{"name": col, "id": col} for col in table_cols]
        
        # Return outputs
        if has_ground_truth:
            return fig, stats_text, cm_display, table_data, table_columns
        else:
            return fig, stats_text, stats_text, table_data, table_columns

else:
    app.layout = html.Div([
        html.H1("Error: No timestamp data available", 
                style={'textAlign':'center', 'color':'#e74c3c'})
    ])

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Starting Advanced Seismic Dashboard v4")
    print("="*80)
    print(f"📍 Access at: http://127.0.0.1:8050")
    print(f"\n✨ Features:")
    print(f"   • Multi-variable operational monitoring with checkboxes")
    print(f"   • Interactive date range selection")
    print(f"   • Searchable & sortable event table (Dash DataTable)")
    print(f"   • Confusion matrix visualization")
    print(f"   • Actual vs predicted comparison toggle")
    print(f"   • Dynamic threshold adjustment")
    print(f"\n📊 Available Operational Variables:")
    for col, label in operational_vars.items():
        if col in df_dashboard.columns:
            print(f"   • {label}")
    print(f"\n📅 Date range: {min_date.date()} to {max_date.date()}" if has_timestamp else "")
    print("="*80 + "\n")
    app.run(host='127.0.0.1', port=8050, debug=True)