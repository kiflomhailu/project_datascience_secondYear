import pandas as pd

print("Saving FULL reverse merge with ALL rows...")
print("This will take a moment (large file)...")

# Load data
seismic = pd.read_csv("seismic_events.csv")
ops = pd.read_csv("operational_metrics.csv")

print(f"Original operational records: {len(ops):,}")

# Convert timestamps
seismic["occurred_at"] = pd.to_datetime(seismic["occurred_at"])
ops["recorded_at"] = pd.to_datetime(ops["recorded_at"])

# Sort
seismic = seismic.sort_values("occurred_at").reset_index(drop=True)
ops = ops.sort_values("recorded_at").reset_index(drop=True)

# Full reverse merge
merged_full = pd.merge_asof(
    left=ops,
    right=seismic,
    left_on="recorded_at",
    right_on="occurred_at",
    direction="nearest",
    tolerance=pd.Timedelta(minutes=5),
    suffixes=("", "_earthquake")
)

print(f"Merged dataset: {len(merged_full):,} rows x {len(merged_full.columns)} columns")
print("\nSaving to: operational_with_earthquakes_FULL.csv")

merged_full.to_csv("operational_with_earthquakes_FULL.csv", index=False)

print("\nDONE!")
print(f"File contains ALL {len(merged_full):,} operational records")
print("File size: ~200-300 MB")

