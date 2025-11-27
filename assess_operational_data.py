import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def assess_operational_data():
    """Comprehensive assessment of operational metrics dataset"""
    
    print("=== OPERATIONAL METRICS DATA ASSESSMENT ===\n")
    
    # Try to read the operational data
    try:
        # Check if file exists and is complete
        import os
        file_path = "Data files and dictionary-20250925T180947Z-1-001/Data files and dictionary/operational_metrics.csv"
        
        if not os.path.exists(file_path):
            print("❌ operational_metrics.csv not found. Please check if download is complete.")
            return
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"📁 File size: {file_size:.1f} MB")
        
        # Read the data
        print("📖 Reading operational data...")
        df_ops = pd.read_csv(file_path)
        
        print(f"✅ Successfully loaded operational data!")
        print(f"📊 Shape: {df_ops.shape} (rows, columns)")
        print(f"📅 Columns: {list(df_ops.columns)}")
        
        # Basic info
        print("\n=== BASIC INFORMATION ===")
        print(f"Total records: {len(df_ops):,}")
        print(f"Total columns: {len(df_ops.columns)}")
        print(f"Memory usage: {df_ops.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Data types
        print("\n=== DATA TYPES ===")
        print(df_ops.dtypes.value_counts())
        
        # Missing values
        print("\n=== MISSING VALUES ===")
        missing = df_ops.isnull().sum()
        missing_pct = (missing / len(df_ops)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        }).sort_values('Missing Count', ascending=False)
        
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # First few rows
        print("\n=== FIRST 5 ROWS ===")
        print(df_ops.head())
        
        # Statistical summary
        print("\n=== STATISTICAL SUMMARY ===")
        print(df_ops.describe())
        
        # Check for timestamp columns
        print("\n=== TIMESTAMP ANALYSIS ===")
        timestamp_cols = [col for col in df_ops.columns if any(word in col.lower() for word in ['time', 'date', 'at', 'timestamp'])]
        print(f"Potential timestamp columns: {timestamp_cols}")
        
        if timestamp_cols:
            for col in timestamp_cols[:2]:  # Check first 2 timestamp columns
                print(f"\n{col} sample values:")
                print(df_ops[col].head())
                print(f"Data type: {df_ops[col].dtype}")
                print(f"Unique values: {df_ops[col].nunique()}")
        
        # Check for operational parameters
        print("\n=== OPERATIONAL PARAMETERS ===")
        op_params = [col for col in df_ops.columns if any(word in col.lower() for word in ['flow', 'pressure', 'temp', 'injection', 'production'])]
        print(f"Operational parameter columns: {op_params}")
        
        if op_params:
            for col in op_params[:5]:  # Show first 5 operational columns
                print(f"\n{col}:")
                print(f"  Range: {df_ops[col].min():.3f} to {df_ops[col].max():.3f}")
                print(f"  Mean: {df_ops[col].mean():.3f}")
                print(f"  Non-null values: {df_ops[col].count():,}")
        
        # Save summary to file
        with open('operational_data_summary.txt', 'w') as f:
            f.write("OPERATIONAL METRICS DATA SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Shape: {df_ops.shape}\n")
            f.write(f"Columns: {list(df_ops.columns)}\n\n")
            f.write("Missing Values:\n")
            f.write(missing_df.to_string())
            f.write("\n\nStatistical Summary:\n")
            f.write(df_ops.describe().to_string())
        
        print(f"\n💾 Summary saved to 'operational_data_summary.txt'")
        
        return df_ops
        
    except Exception as e:
        print(f"❌ Error reading operational data: {str(e)}")
        return None

if __name__ == "__main__":
    df_ops = assess_operational_data()




