import pandas as pd
import numpy as np

"""
REVERSE MERGE: Operational records WITH seismic event flags
Keep ALL 695,625 operational records and mark which ones had earthquakes nearby in time
"""

print("="*70)
print("REVERSE MERGE: OPERATIONAL -> SEISMIC")
print("="*70)
print("Strategy: Keep all operational records, add earthquake info when present")
print()

# Load data
print("Loading data...")
seismic = pd.read_csv("seismic_events.csv")
ops = pd.read_csv("operational_metrics.csv")

print(f"  Seismic events: {len(seismic):,}")
print(f"  Operational records: {len(ops):,}")

# Convert timestamps
seismic["occurred_at"] = pd.to_datetime(seismic["occurred_at"])
ops["recorded_at"] = pd.to_datetime(ops["recorded_at"])

# Sort by time
seismic = seismic.sort_values("occurred_at").reset_index(drop=True)
ops = ops.sort_values("recorded_at").reset_index(drop=True)

# ============================================================================
# REVERSE MERGE: Start with operational data, add seismic info
# ============================================================================
print("\n" + "="*70)
print("PERFORMING REVERSE MERGE...")
print("="*70)

# Key difference: LEFT is operational data, RIGHT is seismic data
tolerance_minutes = 5

merged_reverse = pd.merge_asof(
    left=ops,                    # NOW we keep all operational records
    right=seismic,               # Add seismic data where it matches
    left_on="recorded_at",
    right_on="occurred_at",
    direction="nearest",
    tolerance=pd.Timedelta(minutes=tolerance_minutes),
    suffixes=("", "_earthquake")  # Keep operational names clean
)

print(f"Result: {len(merged_reverse):,} rows (all operational records preserved)")

# ============================================================================
# ANALYSIS: Which operational records had earthquakes?
# ============================================================================
print("\n" + "="*70)
print("EARTHQUAKE DETECTION IN OPERATIONAL DATA")
print("="*70)

# Count how many operational records were near earthquakes
has_earthquake = merged_reverse["occurred_at"].notna().sum()
no_earthquake = merged_reverse["occurred_at"].isna().sum()

print(f"\nOperational records WITH earthquake nearby: {has_earthquake:,}")
print(f"Operational records WITHOUT earthquake:     {no_earthquake:,}")
print(f"Percentage with earthquake: {has_earthquake/len(merged_reverse)*100:.3f}%")

# Calculate time differences for matched records
matched_data = merged_reverse[merged_reverse["occurred_at"].notna()].copy()
matched_data['time_diff_seconds'] = (
    matched_data['occurred_at'] - matched_data['recorded_at']
).dt.total_seconds()

print(f"\nFor matched records:")
print(f"  Average time gap: {matched_data['time_diff_seconds'].mean():.1f} seconds")
print(f"  Max time gap: {matched_data['time_diff_seconds'].abs().max():.1f} seconds")

# ============================================================================
# CREATE EARTHQUAKE FLAG COLUMN
# ============================================================================
print("\n" + "="*70)
print("ADDING EARTHQUAKE FLAG")
print("="*70)

# Add a simple boolean flag
merged_reverse['has_earthquake'] = merged_reverse['occurred_at'].notna()

# Add earthquake magnitude (will be NaN for no earthquake)
merged_reverse['earthquake_magnitude'] = merged_reverse['magnitude']

# Count earthquakes by magnitude
print("\nEarthquake magnitudes in operational data:")
magnitude_counts = merged_reverse[merged_reverse['has_earthquake']]['magnitude'].value_counts().sort_index()
print(f"  Total unique earthquakes matched: {merged_reverse['has_earthquake'].sum():,}")

# ============================================================================
# EXAMPLE: Show some records with and without earthquakes
# ============================================================================
print("\n" + "="*70)
print("SAMPLE DATA")
print("="*70)

print("\nRecords WITH earthquakes (first 5):")
print("-"*70)
with_eq = merged_reverse[merged_reverse['has_earthquake']].head(5)
for idx, row in with_eq.iterrows():
    print(f"  {row['recorded_at']} | Inj flow: {row['inj_flow']:.1f} | "
          f"Earthquake: M{row['magnitude']:.2f}")

print("\nRecords WITHOUT earthquakes (first 5 normal operations):")
print("-"*70)
without_eq = merged_reverse[~merged_reverse['has_earthquake']].head(5)
for idx, row in without_eq.iterrows():
    print(f"  {row['recorded_at']} | Inj flow: {row['inj_flow']:.1f} | "
          f"No earthquake")

# ============================================================================
# SAVE OPTIONS
# ============================================================================
print("\n" + "="*70)
print("SAVE OPTIONS")
print("="*70)

# Option 1: Save full dataset (WARNING: LARGE FILE ~200MB+)
print("\nOption 1: Save ALL operational records with earthquake flags")
print(f"  Size: {len(merged_reverse):,} rows x {len(merged_reverse.columns)} columns")
print("  File size: ~200-300 MB")
print("  Use case: Complete dataset for time series analysis")
save_full = input("\nSave full dataset? (y/n): ").lower().strip()

if save_full == 'y':
    output_full = "operational_with_earthquakes_full.csv"
    merged_reverse.to_csv(output_full, index=False)
    print(f"  Saved: {output_full}")

# Option 2: Save only records near earthquakes (much smaller)
print("\nOption 2: Save ONLY operational records near earthquakes")
earthquake_records = merged_reverse[merged_reverse['has_earthquake']]
print(f"  Size: {len(earthquake_records):,} rows x {len(earthquake_records.columns)} columns")
print(f"  File size: ~1-5 MB")
print("  Use case: Focus on conditions during seismic events")

output_filtered = "operational_during_earthquakes.csv"
earthquake_records.to_csv(output_filtered, index=False)
print(f"  Saved: {output_filtered}")

# Option 3: Create earthquake count per day
print("\nOption 3: Daily earthquake count in operational context")
merged_reverse['date'] = pd.to_datetime(merged_reverse['recorded_at']).dt.date
daily_eq = merged_reverse.groupby('date').agg({
    'has_earthquake': 'sum',
    'inj_flow': 'mean',
    'inj_whp': 'mean',
    'prod_flow': 'mean',
    'prod_temp': 'mean'
}).reset_index()
daily_eq.columns = ['date', 'earthquake_count', 'avg_inj_flow', 'avg_inj_whp', 
                     'avg_prod_flow', 'avg_prod_temp']

output_daily = "daily_operations_with_earthquake_count.csv"
daily_eq.to_csv(output_daily, index=False)
print(f"  Saved: {output_daily}")
print(f"     Size: {len(daily_eq)} days of data")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Original data:
  - {len(ops):,} operational records (every 5 minutes)
  - {len(seismic):,} seismic events

Reverse merge result:
  - {len(merged_reverse):,} total rows (all operational preserved)
  - {has_earthquake:,} records had earthquakes nearby
  - {no_earthquake:,} records had no earthquakes
  
Files created:
  1. {output_filtered} - Only records during earthquakes
  2. {output_daily} - Daily summary with earthquake counts
  3. (Optional) Full dataset if you selected 'y'
  
Use cases:
  - Operational monitoring: Flag dangerous times
  - Time series analysis: Full operational history
  - Pattern detection: When do operations + earthquakes overlap?
""")

print("="*70)
print("DONE!")
print("="*70)

