import os
import pandas as pd

# --- Configuration -------------------------------------------------------
TOLERANCE_MINUTES = 15  # tweak this if you want a wider/narrower match window
OUTPUT_FILENAME = "operational_seismic_merged.csv"

# --- Load data -----------------------------------------------------------
print(" Current working directory:", os.getcwd())

seismic = pd.read_csv("seismic_events.csv")
ops = pd.read_csv("operational_metrics.csv")

print(f" Seismic events loaded: {len(seismic):,}")
print(f" Operational records loaded: {len(ops):,}")

# --- Prepare for merge ---------------------------------------------------
seismic["occurred_at"] = pd.to_datetime(seismic["occurred_at"])
ops["recorded_at"] = pd.to_datetime(ops["recorded_at"])

seismic = seismic.sort_values("occurred_at").reset_index(drop=True)
ops = ops.sort_values("recorded_at").reset_index(drop=True)

tolerance = pd.Timedelta(minutes=TOLERANCE_MINUTES) if TOLERANCE_MINUTES is not None else None

# --- Merge ---------------------------------------------------------------
merged = pd.merge_asof(
    left=seismic,
    right=ops,
    left_on="occurred_at",
    right_on="recorded_at",
    direction="nearest",
    tolerance=tolerance,
    allow_exact_matches=True,
)

# --- Diagnostics ---------------------------------------------------------
matched_rows = merged["recorded_at"].notna().sum()

print(" Merge complete!")
print(f" Output -> {OUTPUT_FILENAME}")
print(f" Rows retained from seismic events: {len(merged):,}")
print(f" Rows matched with operational data: {matched_rows:,}")
if tolerance is not None:
    print(f" Match tolerance: ±{TOLERANCE_MINUTES} minutes")
else:
    print(" Match tolerance: unlimited (nearest record always selected)")

if matched_rows < len(merged):
    print(
        " Some seismic events had no operational record within the tolerance window."
    )

# --- Save ----------------------------------------------------------------
merged.to_csv(OUTPUT_FILENAME, index=False)

print(" Columns in merged dataset:", merged.columns.tolist())