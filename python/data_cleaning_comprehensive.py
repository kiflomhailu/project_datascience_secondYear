"""
Comprehensive Data Cleaning Script for Seismic Events and Operational Metrics

This script provides thorough data cleaning for both datasets:
1. Seismic Events Dataset (seismic_events.csv)
2. Operational Metrics Dataset (operational_metrics.csv)

Features:
- Missing value detection and handling
- Duplicate record identification and removal
- Outlier detection and treatment
- Data type validation and conversion
- Datetime standardization
- Data quality reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self):
        self.cleaning_report = {
            'seismic': {},
            'operational': {}
        }
    
    def load_data(self, seismic_file='seismic_events.csv', operational_file='operational_metrics.csv'):
        """Load both datasets with error handling"""
        print("Loading datasets...")
        
        try:
            # Load seismic data (smaller dataset)
            self.seismic_df = pd.read_csv(seismic_file)
            print(f"✓ Seismic data loaded: {self.seismic_df.shape}")
            
            # Load operational data in chunks due to size
            print("Loading operational data (large file)...")
            chunk_size = 10000
            chunks = []
            
            for chunk in pd.read_csv(operational_file, chunksize=chunk_size):
                chunks.append(chunk)
                if len(chunks) % 10 == 0:
                    print(f"  Loaded {len(chunks) * chunk_size:,} rows...")
            
            self.operational_df = pd.concat(chunks, ignore_index=True)
            print(f"✓ Operational data loaded: {self.operational_df.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def analyze_data_quality(self):
        """Analyze data quality issues in both datasets"""
        print("\n" + "="*60)
        print("DATA QUALITY ANALYSIS")
        print("="*60)
        
        # Analyze seismic data
        print("\n📊 SEISMIC EVENTS DATASET")
        print("-" * 40)
        self._analyze_dataset(self.seismic_df, 'seismic')
        
        # Analyze operational data
        print("\n📊 OPERATIONAL METRICS DATASET")
        print("-" * 40)
        self._analyze_dataset(self.operational_df, 'operational')
    
    def _analyze_dataset(self, df, dataset_name):
        """Analyze individual dataset quality"""
        report = {}
        
        # Basic info
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        if missing.any():
            print(f"\n⚠️  Missing Values:")
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")
            report['missing_values'] = missing[missing > 0].to_dict()
        else:
            print("✓ No missing values found")
            report['missing_values'] = {}
        
        # Duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"\n⚠️  Duplicate rows: {duplicates}")
            report['duplicates'] = duplicates
        else:
            print("✓ No duplicate rows found")
            report['duplicates'] = 0
        
        # Data types
        print(f"\nData Types:")
        for dtype in df.dtypes.value_counts().index:
            count = df.dtypes.value_counts()[dtype]
            print(f"  {dtype}: {count} columns")
        report['data_types'] = df.dtypes.to_dict()
        
        # Datetime columns
        datetime_cols = [col for col in df.columns if 'at' in col.lower() or 'time' in col.lower()]
        if datetime_cols:
            print(f"\nDateTime columns found: {datetime_cols}")
            report['datetime_columns'] = datetime_cols
        
        # Numerical columns analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\nNumerical columns: {len(numerical_cols)}")
            
            # Check for outliers using IQR method
            outliers_info = {}
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    outliers_info[col] = {
                        'count': int(outliers),
                        'percentage': round(outliers / len(df) * 100, 2),
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
            
            if outliers_info:
                print(f"\n⚠️  Potential outliers detected in {len(outliers_info)} columns")
                report['outliers'] = outliers_info
            else:
                print("✓ No significant outliers detected")
                report['outliers'] = {}
        
        self.cleaning_report[dataset_name] = report
    
    def clean_seismic_data(self):
        """Clean seismic events dataset"""
        print("\n" + "="*60)
        print("CLEANING SEISMIC EVENTS DATASET")
        print("="*60)
        
        df = self.seismic_df.copy()
        original_shape = df.shape
        
        # 1. Handle datetime columns
        datetime_cols = ['occurred_at', 'phase_started_at', 'phase_production_ended_at', 'phase_ended_at']
        for col in datetime_cols:
            if col in df.columns:
                print(f"Converting {col} to datetime...")
                df[col] = pd.to_datetime(df[col], errors='coerce')
                invalid_dates = df[col].isnull().sum()
                if invalid_dates > 0:
                    print(f"  ⚠️  {invalid_dates} invalid dates found in {col}")
        
        # 2. Remove duplicates
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        duplicates_removed = duplicates_before - df.duplicated().sum()
        if duplicates_removed > 0:
            print(f"✓ Removed {duplicates_removed} duplicate rows")
        
        # 3. Handle missing values in critical columns
        critical_cols = ['occurred_at', 'magnitude', 'pgv_max']
        for col in critical_cols:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    print(f"⚠️  {missing} missing values in critical column '{col}'")
                    # For this analysis, we'll keep rows but flag them
                    df[f'{col}_missing_flag'] = df[col].isnull()
        
        # 4. Handle outliers in magnitude and pgv_max
        numerical_cols = ['magnitude', 'pgv_max', 'distance_to_fault', 'x', 'y', 'z']
        for col in numerical_cols:
            if col in df.columns and not df[col].isnull().all():
                # Use IQR method but don't remove, just flag
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Use 3*IQR for more conservative outlier detection
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    print(f"⚠️  {outlier_count} potential outliers in {col}")
                    df[f'{col}_outlier_flag'] = outliers
        
        # 5. Validate coordinate consistency
        coordinate_cols = ['x', 'y', 'z']
        if all(col in df.columns for col in coordinate_cols):
            # Check for impossible coordinate values (basic validation)
            coord_issues = 0
            for col in coordinate_cols:
                if col == 'z':  # Depth should typically be negative or zero
                    unusual_z = (df[col] > 1000).sum()  # Unusually high positive z values
                    if unusual_z > 0:
                        print(f"⚠️  {unusual_z} unusual positive z-coordinates (depth values)")
                        coord_issues += unusual_z
            
            if coord_issues == 0:
                print("✓ Coordinate values appear reasonable")
        
        # 6. Create cleaned dataset
        self.seismic_cleaned = df
        
        print(f"\n📋 SEISMIC CLEANING SUMMARY:")
        print(f"  Original shape: {original_shape}")
        print(f"  Cleaned shape:  {df.shape}")
        print(f"  Rows removed:   {original_shape[0] - df.shape[0]}")
        
        return df
    
    def clean_operational_data(self):
        """Clean operational metrics dataset"""
        print("\n" + "="*60)
        print("CLEANING OPERATIONAL METRICS DATASET")
        print("="*60)
        
        df = self.operational_df.copy()
        original_shape = df.shape
        
        print("Processing large dataset in chunks for memory efficiency...")
        
        # 1. Handle datetime columns
        datetime_cols = ['recorded_at', 'phase_started_at', 'phase_production_ended_at', 'phase_ended_at']
        for col in datetime_cols:
            if col in df.columns:
                print(f"Converting {col} to datetime...")
                df[col] = pd.to_datetime(df[col], errors='coerce')
                invalid_dates = df[col].isnull().sum()
                if invalid_dates > 0:
                    print(f"  ⚠️  {invalid_dates} invalid dates found in {col}")
        
        # 2. Remove duplicates
        print("Checking for duplicates...")
        duplicates_before = df.duplicated().sum()
        if duplicates_before > 0:
            df = df.drop_duplicates()
            duplicates_removed = duplicates_before - df.duplicated().sum()
            print(f"✓ Removed {duplicates_removed} duplicate rows")
        else:
            print("✓ No duplicates found")
        
        # 3. Handle missing values in operational columns
        operational_cols = ['inj_flow', 'inj_whp', 'inj_temp', 'prod_temp', 'prod_whp', 'prod_flow']
        
        missing_summary = {}
        for col in operational_cols:
            if col in df.columns:
                missing = df[col].isnull().sum()
                missing_pct = (missing / len(df)) * 100
                missing_summary[col] = {'count': missing, 'percentage': missing_pct}
                
                if missing > 0:
                    print(f"⚠️  {col}: {missing:,} missing values ({missing_pct:.1f}%)")
        
        # 4. Handle negative values in flow and pressure columns
        flow_pressure_cols = ['inj_flow', 'prod_flow', 'inj_whp', 'prod_whp', 'basin_flow']
        for col in flow_pressure_cols:
            if col in df.columns:
                negative_values = (df[col] < 0).sum()
                if negative_values > 0:
                    print(f"⚠️  {negative_values} negative values in {col} (may be valid depending on context)")
                    df[f'{col}_negative_flag'] = df[col] < 0
        
        # 5. Temperature validation
        temp_cols = ['inj_temp', 'prod_temp']
        for col in temp_cols:
            if col in df.columns and not df[col].isnull().all():
                # Flag unreasonable temperatures (outside typical geothermal range)
                unreasonable_temps = ((df[col] < -50) | (df[col] > 400)).sum()
                if unreasonable_temps > 0:
                    print(f"⚠️  {unreasonable_temps} potentially unreasonable temperatures in {col}")
                    df[f'{col}_unreasonable_flag'] = (df[col] < -50) | (df[col] > 400)
        
        # 6. Validate phase consistency
        if 'phase' in df.columns:
            phase_values = df['phase'].value_counts()
            print(f"\nPhase distribution:")
            for phase, count in phase_values.head().items():
                print(f"  Phase {phase}: {count:,} records")
        
        # 7. Create cleaned dataset
        self.operational_cleaned = df
        
        print(f"\n📋 OPERATIONAL CLEANING SUMMARY:")
        print(f"  Original shape: {original_shape}")
        print(f"  Cleaned shape:  {df.shape}")
        print(f"  Rows removed:   {original_shape[0] - df.shape[0]}")
        
        return df
    
    def save_cleaned_data(self):
        """Save cleaned datasets"""
        print("\n" + "="*60)
        print("SAVING CLEANED DATASETS")
        print("="*60)
        
        # Save seismic data
        seismic_output = 'seismic_events_cleaned.csv'
        self.seismic_cleaned.to_csv(seismic_output, index=False)
        print(f"✓ Cleaned seismic data saved as: {seismic_output}")
        
        # Save operational data
        operational_output = 'operational_metrics_cleaned.csv'
        print(f"Saving large operational dataset...")
        self.operational_cleaned.to_csv(operational_output, index=False)
        print(f"✓ Cleaned operational data saved as: {operational_output}")
    
    def generate_cleaning_report(self):
        """Generate comprehensive cleaning report"""
        print("\n" + "="*60)
        print("GENERATING CLEANING REPORT")
        print("="*60)
        
        report_content = []
        report_content.append("DATA CLEANING REPORT")
        report_content.append("=" * 50)
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Seismic dataset summary
        report_content.append("SEISMIC EVENTS DATASET")
        report_content.append("-" * 30)
        seismic_report = self.cleaning_report['seismic']
        report_content.append(f"Original shape: {self.seismic_df.shape}")
        report_content.append(f"Cleaned shape: {self.seismic_cleaned.shape}")
        report_content.append(f"Missing values: {len(seismic_report.get('missing_values', {}))}")
        report_content.append(f"Duplicate rows removed: {seismic_report.get('duplicates', 0)}")
        report_content.append(f"Columns with outliers: {len(seismic_report.get('outliers', {}))}")
        report_content.append("")
        
        # Operational dataset summary
        report_content.append("OPERATIONAL METRICS DATASET")
        report_content.append("-" * 30)
        operational_report = self.cleaning_report['operational']
        report_content.append(f"Original shape: {self.operational_df.shape}")
        report_content.append(f"Cleaned shape: {self.operational_cleaned.shape}")
        report_content.append(f"Missing values: {len(operational_report.get('missing_values', {}))}")
        report_content.append(f"Duplicate rows removed: {operational_report.get('duplicates', 0)}")
        report_content.append(f"Columns with outliers: {len(operational_report.get('outliers', {}))}")
        report_content.append("")
        
        # Detailed findings
        report_content.append("DETAILED FINDINGS")
        report_content.append("-" * 20)
        
        if seismic_report.get('missing_values'):
            report_content.append("\nSeismic Dataset Missing Values:")
            for col, count in seismic_report['missing_values'].items():
                report_content.append(f"  {col}: {count}")
        
        if operational_report.get('missing_values'):
            report_content.append("\nOperational Dataset Missing Values:")
            for col, count in operational_report['missing_values'].items():
                report_content.append(f"  {col}: {count}")
        
        # Save report
        report_filename = 'data_cleaning_report.txt'
        with open(report_filename, 'w') as f:
            f.write('\n'.join(report_content))
        
        print(f"✓ Detailed cleaning report saved as: {report_filename}")
        
        # Print summary
        print("\n📋 CLEANING SUMMARY:")
        print(f"✓ Seismic dataset: {self.seismic_df.shape} → {self.seismic_cleaned.shape}")
        print(f"✓ Operational dataset: {self.operational_df.shape} → {self.operational_cleaned.shape}")
        print(f"✓ Cleaned datasets and report saved")

def main():
    """Main execution function"""
    print("🧹 COMPREHENSIVE DATA CLEANING TOOL")
    print("=" * 60)
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Load data
    cleaner.load_data()
    
    # Analyze data quality
    cleaner.analyze_data_quality()
    
    # Clean datasets
    cleaner.clean_seismic_data()
    cleaner.clean_operational_data()
    
    # Save results
    cleaner.save_cleaned_data()
    cleaner.generate_cleaning_report()
    
    print("\n🎉 DATA CLEANING COMPLETED SUCCESSFULLY!")
    print("\nNext steps:")
    print("1. Review the cleaning report: data_cleaning_report.txt")
    print("2. Use cleaned datasets: seismic_events_cleaned.csv, operational_metrics_cleaned.csv")
    print("3. Consider further domain-specific validation if needed")

if __name__ == "__main__":
    main()


