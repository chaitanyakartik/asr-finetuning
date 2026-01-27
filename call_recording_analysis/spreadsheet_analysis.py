"""
Spreadsheet Analysis Script for AI-ML Call Recordings
Analyzes Excel spreadsheets associated with call recordings
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class SpreadsheetAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.spreadsheets = []
        self.data = {}
        
    def find_all_spreadsheets(self):
        """Find all Excel spreadsheets"""
        xlsx_files = list(self.base_path.rglob("*.xlsx"))
        xls_files = list(self.base_path.rglob("*.xls"))
        csv_files = list(self.base_path.rglob("*.csv"))
        
        all_files = xlsx_files + xls_files + csv_files
        print(f"Found {len(all_files)} spreadsheet files:")
        print(f"  - {len(xlsx_files)} .xlsx files")
        print(f"  - {len(xls_files)} .xls files")
        print(f"  - {len(csv_files)} .csv files")
        
        return all_files
    
    def analyze_spreadsheet(self, file_path):
        """Analyze a single spreadsheet"""
        print(f"\nAnalyzing: {file_path.name}")
        print(f"Department: {file_path.parent.name}")
        
        try:
            # Try reading with different engines
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            
            # Get basic info
            info = {
                'file_path': str(file_path),
                'department': file_path.parent.name,
                'filename': file_path.name,
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': df.columns.tolist(),
                'column_types': df.dtypes.astype(str).to_dict(),
            }
            
            # Print column names
            print(f"  Rows: {info['num_rows']}")
            print(f"  Columns: {info['num_columns']}")
            print(f"  Column names: {info['columns']}")
            
            # Sample data
            print("\n  First few rows:")
            print(df.head(3).to_string())
            
            # Check for common columns related to audio
            audio_related_cols = []
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in 
                       ['call', 'audio', 'recording', 'duration', 'file', 'name', 
                        'id', 'date', 'time', 'phone', 'number']):
                    audio_related_cols.append(col)
            
            if audio_related_cols:
                print(f"\n  Audio-related columns: {audio_related_cols}")
                info['audio_related_columns'] = audio_related_cols
            
            # Store the dataframe
            rel_path = file_path.relative_to(self.base_path)
            self.data[str(rel_path)] = df
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                info['numeric_columns'] = numeric_cols
                info['numeric_summary'] = df[numeric_cols].describe().to_dict()
            
            # Check for missing values
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            missing_info = pd.DataFrame({
                'missing_count': missing,
                'missing_percentage': missing_pct
            })
            missing_info = missing_info[missing_info['missing_count'] > 0]
            if len(missing_info) > 0:
                print("\n  Missing values:")
                print(missing_info.to_string())
                info['missing_values'] = missing_info.to_dict()
            
            return info
            
        except Exception as e:
            print(f"  ERROR: Could not read spreadsheet: {e}")
            return {
                'file_path': str(file_path),
                'department': file_path.parent.name,
                'filename': file_path.name,
                'error': str(e)
            }
    
    def analyze_all_spreadsheets(self):
        """Analyze all spreadsheets"""
        spreadsheet_files = self.find_all_spreadsheets()
        
        results = []
        for file_path in spreadsheet_files:
            info = self.analyze_spreadsheet(file_path)
            results.append(info)
            self.spreadsheets.append(info)
        
        return results
    
    def correlate_with_audio(self, audio_analysis_csv):
        """Correlate spreadsheet data with audio analysis"""
        print("\n" + "="*60)
        print("CORRELATING WITH AUDIO FILES")
        print("="*60)
        
        try:
            audio_df = pd.read_csv(audio_analysis_csv)
            print(f"Loaded {len(audio_df)} audio file records")
            
            correlations = {}
            
            for rel_path, spreadsheet_df in self.data.items():
                department = Path(rel_path).parent.name
                print(f"\nDepartment: {department}")
                
                # Get audio files for this department
                dept_audio = audio_df[audio_df['department'] == department]
                print(f"  Audio files in analysis: {len(dept_audio)}")
                print(f"  Spreadsheet rows: {len(spreadsheet_df)}")
                
                # Try to find matching columns
                # Common patterns: filename might match call ID, recording ID, etc.
                audio_filenames = set(dept_audio['filename'].str.replace('.mp3', '').str.strip())
                
                # Check each column for matches
                matches = {}
                for col in spreadsheet_df.columns:
                    if spreadsheet_df[col].dtype == 'object':  # String columns
                        col_values = set(spreadsheet_df[col].astype(str).str.strip())
                        overlap = audio_filenames & col_values
                        if overlap:
                            matches[col] = len(overlap)
                            print(f"  Column '{col}' has {len(overlap)} matches with audio filenames")
                
                correlations[department] = {
                    'audio_files': len(dept_audio),
                    'spreadsheet_rows': len(spreadsheet_df),
                    'matching_columns': matches
                }
            
            return correlations
            
        except Exception as e:
            print(f"ERROR: Could not correlate data: {e}")
            return {}
    
    def save_results(self, output_dir):
        """Save analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save spreadsheet info
        with open(output_dir / 'spreadsheet_analysis.json', 'w') as f:
            json.dump(self.spreadsheets, f, indent=2, default=str)
        print(f"\nSaved spreadsheet analysis to {output_dir / 'spreadsheet_analysis.json'}")
        
        # Save each spreadsheet as CSV for easy access
        csv_dir = output_dir / 'spreadsheets_csv'
        csv_dir.mkdir(exist_ok=True)
        
        for rel_path, df in self.data.items():
            safe_name = rel_path.replace('/', '_').replace(' ', '_')
            csv_path = csv_dir / f"{safe_name}.csv"
            df.to_csv(csv_path, index=False)
        
        print(f"Saved {len(self.data)} spreadsheets as CSV to {csv_dir}")


def main():
    base_path = "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML"
    output_dir = "/Users/chaitanyakartik/Projects/asr-finetuning/call_recording_analysis"
    
    print("="*60)
    print("Call Recording Spreadsheet Analysis")
    print("="*60)
    
    analyzer = SpreadsheetAnalyzer(base_path)
    
    # Analyze all spreadsheets
    results = analyzer.analyze_all_spreadsheets()
    
    # Save results
    analyzer.save_results(output_dir)
    
    # Try to correlate with audio analysis if available
    audio_csv = Path(output_dir) / 'detailed_audio_analysis.csv'
    if audio_csv.exists():
        correlations = analyzer.correlate_with_audio(audio_csv)
        
        # Save correlation results
        with open(Path(output_dir) / 'audio_spreadsheet_correlation.json', 'w') as f:
            json.dump(correlations, f, indent=2)
        print(f"\nSaved correlation results to {Path(output_dir) / 'audio_spreadsheet_correlation.json'}")
    else:
        print("\nNo audio analysis CSV found. Run audio_analysis.py first to enable correlation.")


if __name__ == "__main__":
    main()
