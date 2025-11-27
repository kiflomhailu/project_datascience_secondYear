import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Read the seismic data
df = pd.read_csv(r"C:/Users/Tech/Desktop/YEAR_SECOND/Data files and dictionary-20250925T180947Z-1-001/Data files and dictionary/seismic_events.csv")


# Convert date columns to datetime
df['occurred_at'] = pd.to_datetime(df['occurred_at'])
df['phase_started_at'] = pd.to_datetime(df['phase_started_at'])
df['phase_ended_at'] = pd.to_datetime(df['phase_ended_at'])

# # Create the main graph for Page 8
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# # Adjust layout to prevent overlap
# plt.subplots_adjust(hspace=0.35, top=0.90, bottom=0.1)

# Create the main graph for Page 8
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Adjust layout to prevent overlap
plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.90, bottom=0.1)

# Add main title (moved slightly down)
fig.suptitle('Balmatt Geothermal Plant - Seismic Monitoring Dashboard',
             fontsize=18, fontweight='bold', y=0.95)


# Graph 1: PGV Max over time with production periods
ax1.plot(df['occurred_at'], df['pgv_max'], 'o-', markersize=3, linewidth=1, alpha=0.7, color='blue')
ax1.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Alert Threshold (0.2)')

# Highlight production periods
for idx, row in df.iterrows():
    if row['is_producing']:
        ax1.axvspan(row['phase_started_at'], row['phase_ended_at'], alpha=0.2, color='green', label='Producing Period' if idx == 0 else "")

# Annotate peak events
peak_events = df[df['pgv_max'] > 0.2].nlargest(3, 'pgv_max')
for i, (idx, row) in enumerate(peak_events.iterrows()):
    # Adjust annotation position to avoid overlap
    offset_x = 20 if i % 2 == 0 else -20
    offset_y = 20 if i < 2 else -30
    ax1.annotate(f'PGV: {row["pgv_max"]:.2f}\nDate: {row["occurred_at"].strftime("%Y-%m-%d")}', 
                xy=(row['occurred_at'], row['pgv_max']), 
                xytext=(offset_x, offset_y), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
                fontsize=9)

ax1.set_ylabel('Peak Ground Velocity (m/s)')
ax1.set_title('Seismic Activity Over Time - Peak Ground Velocity', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graph 2: Daily event count
daily_events = df.groupby(df['occurred_at'].dt.date).size()
ax2.bar(daily_events.index, daily_events.values, alpha=0.7, color='orange')
ax2.set_ylabel('Number of Events')
ax2.set_title('Daily Seismic Event Count', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Format x-axis for both plots
for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=10)

# Add main title (moved down a bit)
fig.suptitle('Balmatt Geothermal Plant - Seismic Monitoring Dashboard',
             fontsize=18, fontweight='bold', y=0.96)

# Save and show
plt.savefig('seismic_analysis_page8.png', dpi=300, bbox_inches='tight')
plt.show()

# Print key statistics for the slide
print("=== KEY STATISTICS FOR PAGE 8 ===")
print(f"Total events: {len(df)}")
print(f"Date range: {df['occurred_at'].min().strftime('%Y-%m-%d')} to {df['occurred_at'].max().strftime('%Y-%m-%d')}")
print(f"Magnitude range: {df['magnitude'].min():.2f} to {df['magnitude'].max():.2f}")
print(f"PGV max range: {df['pgv_max'].min():.3f} to {df['pgv_max'].max():.3f} m/s")
print(f"Events during production: {df['is_producing'].sum()} ({df['is_producing'].mean()*100:.1f}%)")
print(f"Events above alert threshold (PGV > 0.2): {(df['pgv_max'] > 0.2).sum()}")
print(f"Average events per day: {len(df) / ((df['occurred_at'].max() - df['occurred_at'].min()).days):.1f}")

# Top 5 highest PGV events
print("\n=== TOP 5 HIGHEST PGV EVENTS ===")
top_events = df.nlargest(5, 'pgv_max')[['occurred_at', 'pgv_max', 'magnitude', 'is_producing']]
for idx, row in top_events.iterrows():
    print(f"{row['occurred_at'].strftime('%Y-%m-%d')}: PGV={row['pgv_max']:.3f}, Mag={row['magnitude']:.2f}, Producing={row['is_producing']}")
