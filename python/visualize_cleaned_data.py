"""
Data Visualization Script for Cleaned Datasets

This script creates visualizations to help you understand your cleaned data:
- Data quality overview
- Missing value patterns
- Outlier distribution
- Time series patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_data_quality_overview():
    """Create overview visualizations of data quality"""
    
    # Load cleaned datasets
    print("Loading cleaned datasets...")
    seismic_df = pd.read_csv('seismic_events_cleaned.csv')
    
    # Load operational data (sample for visualization)
    operational_sample = pd.read_csv('operational_metrics_cleaned.csv', nrows=10000)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Quality Overview - Cleaned Datasets', fontsize=16, fontweight='bold')
    
    # 1. Missing values heatmap for operational data (sample)
    operational_cols = ['inj_flow', 'inj_whp', 'inj_temp', 'prod_temp', 'prod_whp', 'prod_flow']
    missing_data = operational_sample[operational_cols].isnull()
    
    sns.heatmap(missing_data, cbar=True, yticklabels=False, 
                cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('Missing Values Pattern (Sample)\nOperational Data')
    axes[0,0].set_xlabel('Columns')
    
    # 2. Seismic events magnitude distribution
    seismic_df['magnitude_numeric'] = pd.to_numeric(seismic_df['magnitude'], errors='coerce')
    axes[0,1].hist(seismic_df['magnitude_numeric'].dropna(), bins=20, 
                   color='skyblue', alpha=0.7, edgecolor='black')
    axes[0,1].set_title('Seismic Events - Magnitude Distribution')
    axes[0,1].set_xlabel('Magnitude')
    axes[0,1].set_ylabel('Frequency')
    
    # 3. Operational data - Flow rates over time (sample)
    operational_sample['recorded_at'] = pd.to_datetime(operational_sample['recorded_at'])
    sample_data = operational_sample.sort_values('recorded_at').head(1000)
    
    axes[1,0].plot(sample_data['recorded_at'], sample_data['inj_flow'], 
                   alpha=0.6, label='Injection Flow', linewidth=1)
    axes[1,0].plot(sample_data['recorded_at'], sample_data['prod_flow'], 
                   alpha=0.6, label='Production Flow', linewidth=1)
    axes[1,0].set_title('Flow Rates Over Time (Sample)')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Flow Rate')
    axes[1,0].legend()
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Data completeness summary
    seismic_completeness = (1 - seismic_df.isnull().sum() / len(seismic_df)) * 100
    operational_completeness = (1 - operational_sample[operational_cols].isnull().sum() / len(operational_sample)) * 100
    
    completeness_data = pd.DataFrame({
        'Seismic': seismic_completeness[:6],  # First 6 columns
        'Operational': operational_completeness
    }).fillna(100)
    
    completeness_data.plot(kind='bar', ax=axes[1,1], color=['lightcoral', 'lightblue'])
    axes[1,1].set_title('Data Completeness by Column')
    axes[1,1].set_ylabel('Completeness (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('data_quality_overview.png', dpi=300, bbox_inches='tight')
    print("✓ Data quality overview saved as: data_quality_overview.png")
    
    return fig

def create_missing_values_summary():
    """Create detailed missing values analysis"""
    
    # Load operational data for missing value analysis
    print("Analyzing missing values patterns...")
    
    # Read in chunks to get accurate missing value counts
    chunk_size = 50000
    missing_counts = {}
    total_rows = 0
    
    for chunk in pd.read_csv('operational_metrics_cleaned.csv', chunksize=chunk_size):
        total_rows += len(chunk)
        chunk_missing = chunk.isnull().sum()
        
        for col in chunk_missing.index:
            if col not in missing_counts:
                missing_counts[col] = 0
            missing_counts[col] += chunk_missing[col]
    
    # Calculate percentages
    missing_percentages = {col: (count / total_rows) * 100 
                          for col, count in missing_counts.items() 
                          if count > 0}
    
    if missing_percentages:
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        cols = list(missing_percentages.keys())
        percentages = list(missing_percentages.values())
        
        bars = ax.bar(range(len(cols)), percentages, color='salmon', alpha=0.7)
        ax.set_xlabel('Columns')
        ax.set_ylabel('Missing Percentage (%)')
        ax.set_title('Missing Values by Column - Operational Dataset')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Missing values analysis saved as: missing_values_analysis.png")
        
        return missing_percentages
    else:
        print("✓ No missing values found in dataset")
        return {}

def create_summary_statistics():
    """Generate summary statistics for cleaned datasets"""
    
    print("Generating summary statistics...")
    
    # Seismic data summary
    seismic_df = pd.read_csv('seismic_events_cleaned.csv')
    
    # Operational data summary (sample)
    operational_sample = pd.read_csv('operational_metrics_cleaned.csv', nrows=10000)
    
    summary_report = []
    summary_report.append("CLEANED DATASETS - SUMMARY STATISTICS")
    summary_report.append("=" * 50)
    summary_report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_report.append("")
    
    # Seismic summary
    summary_report.append("SEISMIC EVENTS DATASET")
    summary_report.append("-" * 30)
    summary_report.append(f"Total events: {len(seismic_df):,}")
    
    if 'magnitude' in seismic_df.columns:
        mag_numeric = pd.to_numeric(seismic_df['magnitude'], errors='coerce')
        summary_report.append(f"Magnitude range: {mag_numeric.min():.2f} to {mag_numeric.max():.2f}")
        summary_report.append(f"Average magnitude: {mag_numeric.mean():.2f}")
    
    if 'occurred_at' in seismic_df.columns:
        seismic_df['occurred_at'] = pd.to_datetime(seismic_df['occurred_at'])
        date_range = seismic_df['occurred_at'].max() - seismic_df['occurred_at'].min()
        summary_report.append(f"Time span: {date_range.days} days")
        summary_report.append(f"Date range: {seismic_df['occurred_at'].min().date()} to {seismic_df['occurred_at'].max().date()}")
    
    summary_report.append("")
    
    # Operational summary
    summary_report.append("OPERATIONAL METRICS DATASET (Sample)")
    summary_report.append("-" * 30)
    summary_report.append(f"Sample size: {len(operational_sample):,}")
    
    if 'recorded_at' in operational_sample.columns:
        operational_sample['recorded_at'] = pd.to_datetime(operational_sample['recorded_at'])
        date_range = operational_sample['recorded_at'].max() - operational_sample['recorded_at'].min()
        summary_report.append(f"Sample time span: {date_range.days} days")
    
    # Key operational metrics
    key_metrics = ['inj_flow', 'prod_flow', 'inj_temp', 'prod_temp']
    summary_report.append("\nKey Metrics Summary:")
    for metric in key_metrics:
        if metric in operational_sample.columns:
            values = operational_sample[metric].dropna()
            if len(values) > 0:
                summary_report.append(f"  {metric}: mean={values.mean():.2f}, std={values.std():.2f}")
    
    # Save summary
    with open('cleaned_data_summary.txt', 'w') as f:
        f.write('\n'.join(summary_report))
    
    print("✓ Summary statistics saved as: cleaned_data_summary.txt")
    
    return summary_report

def main():
    """Main execution function"""
    print("📊 VISUALIZING CLEANED DATA")
    print("=" * 50)
    
    try:
        # Create visualizations
        create_data_quality_overview()
        missing_stats = create_missing_values_summary()
        summary_stats = create_summary_statistics()
        
        print("\n🎉 VISUALIZATION COMPLETED!")
        print("\nFiles created:")
        print("✓ data_quality_overview.png - Overall data quality visualization")
        if missing_stats:
            print("✓ missing_values_analysis.png - Missing values breakdown")
        print("✓ cleaned_data_summary.txt - Statistical summary")
        
        print("\n📋 QUICK INSIGHTS:")
        if missing_stats:
            highest_missing = max(missing_stats.items(), key=lambda x: x[1])
            print(f"• Highest missing data: {highest_missing[0]} ({highest_missing[1]:.1f}%)")
            print(f"• Columns with missing data: {len(missing_stats)}")
        else:
            print("• No missing values in operational data")
        
        print("• Both datasets are now analysis-ready!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Please ensure the cleaned CSV files exist in the current directory.")

if __name__ == "__main__":
    main()


