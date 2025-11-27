import pandas as pd
import numpy as np

"""
IMPROVED MERGE STRATEGY FOR SEISMIC + OPERATIONAL DATA
Based on timestamp analysis and data characteristics
"""

print("="*70)
print("IMPROVED SEISMIC-OPERATIONAL MERGE")
print("="*70)

# Load data
seismic = pd.read_csv("seismic_events.csv")
ops = pd.read_csv("operational_metrics.csv")

print(f"\nLoaded:")
print(f"  - {len(seismic):,} seismic events")
print(f"  - {len(ops):,} operational records (every 5 minutes)")

# Convert timestamps
seismic["occurred_at"] = pd.to_datetime(seismic["occurred_at"])
ops["recorded_at"] = pd.to_datetime(ops["recorded_at"])

# Sort by time
seismic = seismic.sort_values("occurred_at").reset_index(drop=True)
ops = ops.sort_values("recorded_at").reset_index(drop=True)

print(f"\nSeismic events: {seismic['occurred_at'].min()} to {seismic['occurred_at'].max()}")
print(f"Operational data: {ops['recorded_at'].min()} to {ops['recorded_at'].max()}")

# ============================================================================
# RECOMMENDATION 1: Use 5-minute tolerance (matches operational sampling rate)
# ============================================================================
print("\n" + "="*70)
print("STRATEGY 1: EXACT NEAREST MATCH (5-min tolerance)")
print("="*70)
print("Best for: Understanding exact conditions when earthquake occurred")

tolerance_minutes = 5  # Match operational sampling interval

merged_exact = pd.merge_asof(
    left=seismic,
    right=ops,
    left_on="occurred_at",
    right_on="recorded_at",
    direction="nearest",
    tolerance=pd.Timedelta(minutes=tolerance_minutes),
    suffixes=("_seismic", "_operational")  # Better naming than _x/_y
)

matched = merged_exact["recorded_at"].notna().sum()
print(f"  Matched: {matched}/{len(merged_exact)} events ({matched/len(merged_exact)*100:.1f}%)")
print(f"  Tolerance: ±{tolerance_minutes} minutes")
print(f"  Unmatched: {len(merged_exact) - matched} events")

# Calculate time differences
merged_exact['time_diff_seconds'] = (
    merged_exact['occurred_at'] - merged_exact['recorded_at']
).dt.total_seconds()

print(f"\n  Time gap statistics:")
print(f"    Average: {merged_exact['time_diff_seconds'].mean():.1f} seconds")
print(f"    Max: {merged_exact['time_diff_seconds'].abs().max():.1f} seconds")
print(f"    Within 1 min: {(merged_exact['time_diff_seconds'].abs() <= 60).sum()}/{matched}")

# ============================================================================
# RECOMMENDATION 2: Get conditions BEFORE the earthquake
# ============================================================================
print("\n" + "="*70)
print("STRATEGY 2: CONDITIONS BEFORE EARTHQUAKE (backward direction)")
print("="*70)
print("Best for: What were conditions leading up to the earthquake?")

merged_before = pd.merge_asof(
    left=seismic,
    right=ops,
    left_on="occurred_at",
    right_on="recorded_at",
    direction="backward",  # Get most recent operational record BEFORE event
    tolerance=pd.Timedelta(minutes=30),  # Look back up to 30 minutes
    suffixes=("_seismic", "_operational")
)

matched_before = merged_before["recorded_at"].notna().sum()
print(f"  Matched: {matched_before}/{len(merged_before)} events ({matched_before/len(merged_before)*100:.1f}%)")
print(f"  Direction: backward (captures conditions leading up to event)")

# ============================================================================
# RECOMMENDATION 3: Time window aggregation
# ============================================================================
print("\n" + "="*70)
print("STRATEGY 3: AGGREGATED TIME WINDOW (1-hour before each event)")
print("="*70)
print("Best for: Understanding patterns/trends before earthquakes")

def get_pre_event_stats(seismic_event, ops_data, hours_back=1):
    """Get average operational conditions in X hours before earthquake"""
    event_time = seismic_event['occurred_at']
    window_start = event_time - pd.Timedelta(hours=hours_back)
    
    # Filter operational data in time window
    window_data = ops_data[
        (ops_data['recorded_at'] >= window_start) & 
        (ops_data['recorded_at'] < event_time)
    ]
    
    if len(window_data) == 0:
        return None
    
    # Calculate statistics
    stats = {
        'event_id': seismic_event['event_id'],
        'occurred_at': event_time,
        'n_records': len(window_data),
        'avg_inj_flow': window_data['inj_flow'].mean(),
        'max_inj_flow': window_data['inj_flow'].max(),
        'avg_inj_whp': window_data['inj_whp'].mean(),
        'max_inj_whp': window_data['inj_whp'].max(),
        'avg_prod_temp': window_data['prod_temp'].mean(),
        'avg_prod_flow': window_data['prod_flow'].mean(),
    }
    return stats

# Example: Calculate for first 5 events
print("  Example for first 5 events (1-hour window):")
print("  " + "-"*60)
for idx in range(min(5, len(seismic))):
    stats = get_pre_event_stats(seismic.iloc[idx], ops, hours_back=1)
    if stats:
        print(f"  Event {stats['event_id']}: {stats['n_records']} records, "
              f"avg inj_flow={stats['avg_inj_flow']:.2f}")
    else:
        print(f"  Event {seismic.iloc[idx]['event_id']}: No data in window")

# ============================================================================
# SAVE RECOMMENDED MERGE
# ============================================================================
print("\n" + "="*70)
print("SAVING RECOMMENDED MERGE")
print("="*70)

output_file = "seismic_operational_improved.csv"
merged_exact.to_csv(output_file, index=False)
print(f"  Saved: {output_file}")
print(f"  Rows: {len(merged_exact)}")
print(f"  Columns: {len(merged_exact.columns)}")

# Save column reference
print(f"\n  Key columns in merged dataset:")
print(f"    Seismic: occurred_at, event_id, magnitude, pgv_max, location")
print(f"    Operational: recorded_at, inj_flow, inj_whp, inj_temp, prod_flow, prod_temp")
print(f"    Time tracking: time_diff_seconds")

print("\n" + "="*70)
print("DONE!")
print("="*70)

