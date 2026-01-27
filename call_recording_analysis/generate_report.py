"""
Generate comprehensive analysis report with visualizations
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ReportGenerator:
    def __init__(self, analysis_dir):
        self.analysis_dir = Path(analysis_dir)
        self.output_dir = self.analysis_dir / 'reports'
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load all analysis data"""
        print("Loading analysis data...")
        
        # Load audio analysis
        audio_csv = self.analysis_dir / 'detailed_audio_analysis.csv'
        if audio_csv.exists():
            self.audio_df = pd.read_csv(audio_csv)
            print(f"  Loaded {len(self.audio_df)} audio records")
        else:
            self.audio_df = None
            print("  No audio analysis found")
        
        # Load summary stats
        summary_json = self.analysis_dir / 'summary_stats.json'
        if summary_json.exists():
            with open(summary_json, 'r') as f:
                self.summary = json.load(f)
            print(f"  Loaded summary statistics")
        else:
            self.summary = None
            print("  No summary stats found")
        
        # Load spreadsheet analysis
        spreadsheet_json = self.analysis_dir / 'spreadsheet_analysis.json'
        if spreadsheet_json.exists():
            with open(spreadsheet_json, 'r') as f:
                self.spreadsheet_info = json.load(f)
            print(f"  Loaded {len(self.spreadsheet_info)} spreadsheet records")
        else:
            self.spreadsheet_info = None
            print("  No spreadsheet analysis found")
    
    def create_visualizations(self):
        """Create all visualizations"""
        if self.audio_df is None:
            print("Cannot create visualizations without audio data")
            return
        
        print("\nCreating visualizations...")
        
        # 1. Files per department
        self.plot_files_per_department()
        
        # 2. Duration distribution
        self.plot_duration_distribution()
        
        # 3. Sample rate distribution
        self.plot_sample_rate_distribution()
        
        # 4. File size distribution
        self.plot_file_size_distribution()
        
        # 5. Department comparison
        self.plot_department_comparison()
        
        # 6. Duration by department (boxplot)
        self.plot_duration_by_department()
        
        print(f"All visualizations saved to {self.output_dir}")
    
    def plot_files_per_department(self):
        """Plot number of files per department"""
        plt.figure(figsize=(14, 8))
        dept_counts = self.audio_df['department'].value_counts().sort_values(ascending=True)
        
        ax = dept_counts.plot(kind='barh', color='steelblue')
        plt.title('Number of Audio Files per Department', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Files', fontsize=12)
        plt.ylabel('Department', fontsize=12)
        plt.tight_layout()
        
        # Add value labels
        for i, v in enumerate(dept_counts):
            ax.text(v + 1, i, str(v), va='center', fontsize=9)
        
        plt.savefig(self.output_dir / 'files_per_department.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Files per department")
    
    def plot_duration_distribution(self):
        """Plot duration distribution"""
        if 'duration_seconds' not in self.audio_df.columns or self.audio_df['duration_seconds'].isna().all():
            print("  ✗ Skipping duration distribution (no data)")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        durations_min = self.audio_df['duration_seconds'] / 60
        axes[0].hist(durations_min, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Duration (minutes)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Call Durations', fontsize=14, fontweight='bold')
        axes[0].axvline(durations_min.mean(), color='red', linestyle='--', 
                       label=f'Mean: {durations_min.mean():.2f} min')
        axes[0].axvline(durations_min.median(), color='green', linestyle='--', 
                       label=f'Median: {durations_min.median():.2f} min')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(durations_min, vert=True)
        axes[1].set_ylabel('Duration (minutes)', fontsize=12)
        axes[1].set_title('Duration Statistics (Boxplot)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'duration_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Duration distribution")
    
    def plot_sample_rate_distribution(self):
        """Plot sample rate distribution"""
        if 'sample_rate' not in self.audio_df.columns or self.audio_df['sample_rate'].isna().all():
            print("  ✗ Skipping sample rate distribution (no data)")
            return
        
        plt.figure(figsize=(10, 6))
        sample_rate_counts = self.audio_df['sample_rate'].value_counts().sort_index()
        
        ax = sample_rate_counts.plot(kind='bar', color='coral', edgecolor='black', alpha=0.7)
        plt.title('Sample Rate Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Sample Rate (Hz)', fontsize=12)
        plt.ylabel('Number of Files', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add value labels
        for i, v in enumerate(sample_rate_counts):
            ax.text(i, v + 1, str(v), ha='center', fontsize=9)
        
        plt.savefig(self.output_dir / 'sample_rate_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Sample rate distribution")
    
    def plot_file_size_distribution(self):
        """Plot file size distribution"""
        if 'file_size_mb' not in self.audio_df.columns:
            print("  ✗ Skipping file size distribution (no data)")
            return
        
        plt.figure(figsize=(12, 6))
        plt.hist(self.audio_df['file_size_mb'], bins=50, color='lightgreen', 
                edgecolor='black', alpha=0.7)
        plt.xlabel('File Size (MB)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of File Sizes', fontsize=16, fontweight='bold')
        plt.axvline(self.audio_df['file_size_mb'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.audio_df["file_size_mb"].mean():.2f} MB')
        plt.axvline(self.audio_df['file_size_mb'].median(), color='green', linestyle='--', 
                   label=f'Median: {self.audio_df["file_size_mb"].median():.2f} MB')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'file_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ File size distribution")
    
    def plot_department_comparison(self):
        """Plot comparison of departments"""
        if self.summary is None or 'department_stats' not in self.summary:
            print("  ✗ Skipping department comparison (no data)")
            return
        
        dept_df = pd.DataFrame(self.summary['department_stats']).T
        dept_df = dept_df.sort_values('file_count', ascending=False).head(15)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # File count
        dept_df['file_count'].plot(kind='bar', ax=axes[0, 0], color='steelblue', alpha=0.7)
        axes[0, 0].set_title('File Count by Department (Top 15)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Files')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total duration
        if 'total_duration_hours' in dept_df.columns:
            dept_df['total_duration_hours'].plot(kind='bar', ax=axes[0, 1], color='coral', alpha=0.7)
            axes[0, 1].set_title('Total Duration by Department (Top 15)', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Duration (hours)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average duration
        if 'avg_duration_seconds' in dept_df.columns:
            (dept_df['avg_duration_seconds'] / 60).plot(kind='bar', ax=axes[1, 0], 
                                                          color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Average Call Duration by Department (Top 15)', 
                                fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Duration (minutes)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Total size
        if 'total_size_mb' in dept_df.columns:
            (dept_df['total_size_mb'] / 1024).plot(kind='bar', ax=axes[1, 1], 
                                                    color='plum', alpha=0.7)
            axes[1, 1].set_title('Total Data Size by Department (Top 15)', 
                                fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Size (GB)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'department_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Department comparison")
    
    def plot_duration_by_department(self):
        """Plot duration distribution by department"""
        if 'duration_seconds' not in self.audio_df.columns or self.audio_df['duration_seconds'].isna().all():
            print("  ✗ Skipping duration by department (no data)")
            return
        
        # Get top departments by file count
        top_depts = self.audio_df['department'].value_counts().head(10).index.tolist()
        df_subset = self.audio_df[self.audio_df['department'].isin(top_depts)].copy()
        df_subset['duration_minutes'] = df_subset['duration_seconds'] / 60
        
        plt.figure(figsize=(14, 8))
        
        # Create boxplot
        df_subset.boxplot(column='duration_minutes', by='department', 
                         figsize=(14, 8), rot=45)
        plt.title('Call Duration Distribution by Department (Top 10)', 
                 fontsize=16, fontweight='bold')
        plt.suptitle('')  # Remove automatic title
        plt.ylabel('Duration (minutes)', fontsize=12)
        plt.xlabel('Department', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'duration_by_department_boxplot.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Duration by department boxplot")
    
    def generate_markdown_report(self):
        """Generate a comprehensive markdown report"""
        print("\nGenerating markdown report...")
        
        report_path = self.output_dir / 'analysis_report.md'
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# Call Recording Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            if self.summary:
                f.write(f"- **Total Audio Files:** {self.summary.get('total_files', 'N/A')}\n")
                f.write(f"- **Total Departments:** {self.summary.get('total_departments', 'N/A')}\n")
                if 'total_duration_hours' in self.summary:
                    f.write(f"- **Total Duration:** {self.summary['total_duration_hours']:.2f} hours "
                           f"({self.summary['total_duration_hours']/24:.2f} days)\n")
                    f.write(f"- **Average Call Duration:** {self.summary['avg_duration_seconds']:.2f} seconds "
                           f"({self.summary['avg_duration_seconds']/60:.2f} minutes)\n")
                    f.write(f"- **Median Call Duration:** {self.summary['median_duration_seconds']:.2f} seconds\n")
                if 'total_size_gb' in self.summary:
                    f.write(f"- **Total Data Size:** {self.summary['total_size_gb']:.2f} GB\n")
                if 'most_common_sample_rate' in self.summary:
                    f.write(f"- **Most Common Sample Rate:** {self.summary['most_common_sample_rate']} Hz\n")
            f.write("\n")
            
            # Audio Quality Metrics
            f.write("## Audio Quality Metrics\n\n")
            if self.summary and 'sample_rates' in self.summary:
                f.write("### Sample Rate Distribution\n\n")
                f.write("| Sample Rate (Hz) | Count |\n")
                f.write("|------------------|-------|\n")
                for sr, count in sorted(self.summary['sample_rates'].items()):
                    f.write(f"| {sr} | {count} |\n")
                f.write("\n")
            
            if self.summary and 'channel_distribution' in self.summary:
                f.write("### Channel Distribution\n\n")
                f.write("| Channels | Count |\n")
                f.write("|----------|-------|\n")
                for ch, count in sorted(self.summary['channel_distribution'].items()):
                    ch_str = 'Mono' if int(ch) == 1 else 'Stereo' if int(ch) == 2 else f'{ch} channels'
                    f.write(f"| {ch_str} | {count} |\n")
                f.write("\n")
            
            # Department Statistics
            f.write("## Department Statistics\n\n")
            if self.summary and 'department_stats' in self.summary:
                dept_df = pd.DataFrame(self.summary['department_stats']).T
                dept_df = dept_df.sort_values('file_count', ascending=False)
                
                f.write("### Complete Department Breakdown\n\n")
                f.write("| Department | Files | % of Total | ")
                if 'total_duration_hours' in dept_df.columns:
                    f.write("Total Duration (hrs) | Avg Duration (min) | Total Size (MB) |")
                f.write("\n")
                f.write("|------------|-------|------------|")
                if 'total_duration_hours' in dept_df.columns:
                    f.write("---------------------|-------------------|----------------|")
                f.write("\n")
                
                for dept, row in dept_df.iterrows():
                    f.write(f"| {dept} | {row['file_count']} | {row['percentage']:.1f}% | ")
                    if 'total_duration_hours' in row:
                        f.write(f"{row['total_duration_hours']:.2f} | "
                               f"{row['avg_duration_seconds']/60:.2f} | "
                               f"{row['total_size_mb']:.2f} |")
                    f.write("\n")
                f.write("\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            
            viz_files = [
                ('files_per_department.png', 'Files per Department'),
                ('duration_distribution.png', 'Duration Distribution'),
                ('sample_rate_distribution.png', 'Sample Rate Distribution'),
                ('file_size_distribution.png', 'File Size Distribution'),
                ('department_comparison.png', 'Department Comparison'),
                ('duration_by_department_boxplot.png', 'Duration by Department'),
            ]
            
            for viz_file, title in viz_files:
                viz_path = self.output_dir / viz_file
                if viz_path.exists():
                    f.write(f"### {title}\n\n")
                    f.write(f"![{title}]({viz_file})\n\n")
            
            # Data Quality Notes
            f.write("## Data Quality & Recommendations\n\n")
            f.write("### For ASR Training/Testing\n\n")
            
            if self.summary and 'sample_rates' in self.summary:
                rates = list(self.summary['sample_rates'].keys())
                if len(rates) > 1:
                    f.write("⚠️ **Multiple Sample Rates Detected:**\n")
                    f.write("- Recommendation: Resample all audio to a consistent rate "
                           f"(preferably {self.summary.get('most_common_sample_rate', 16000)} Hz)\n\n")
                else:
                    f.write("✅ **Consistent Sample Rate:**\n")
                    f.write(f"- All audio at {rates[0]} Hz - Good for ASR training\n\n")
            
            if self.summary and 'avg_duration_seconds' in self.summary:
                avg_dur = self.summary['avg_duration_seconds']
                if avg_dur < 3:
                    f.write("⚠️ **Very Short Average Duration:**\n")
                    f.write("- Very short clips may be challenging for ASR\n")
                    f.write("- Consider filtering or combining segments\n\n")
                elif avg_dur > 300:  # 5 minutes
                    f.write("⚠️ **Long Average Duration:**\n")
                    f.write("- Long recordings may need segmentation\n")
                    f.write("- Consider chunking for training\n\n")
                else:
                    f.write("✅ **Good Duration Range for ASR:**\n")
                    f.write(f"- Average {avg_dur:.1f}s is suitable for ASR training\n\n")
            
            f.write("### Next Steps\n\n")
            f.write("1. **Transcription:**\n")
            f.write("   - Identify which departments have metadata/transcriptions\n")
            f.write("   - Consider using Whisper or similar for automatic transcription\n")
            f.write("   - Human verification recommended for quality\n\n")
            
            f.write("2. **Data Preprocessing:**\n")
            f.write("   - Normalize audio levels\n")
            f.write("   - Resample to consistent rate (16kHz recommended)\n")
            f.write("   - Convert to mono if stereo\n")
            f.write("   - Remove silence/noise if needed\n\n")
            
            f.write("3. **Dataset Preparation:**\n")
            f.write("   - Create train/val/test splits by department\n")
            f.write("   - Balance dataset if departments vary significantly\n")
            f.write("   - Create manifest files for NeMo or similar frameworks\n\n")
            
            f.write("4. **Quality Assessment:**\n")
            f.write("   - Manual review of sample recordings\n")
            f.write("   - Check for background noise, multiple speakers\n")
            f.write("   - Assess language/dialect consistency\n\n")
            
            # Spreadsheet Info
            if self.spreadsheet_info:
                f.write("## Spreadsheet Data\n\n")
                f.write(f"Found {len(self.spreadsheet_info)} spreadsheet files across departments.\n\n")
                f.write("### Spreadsheet Summary\n\n")
                f.write("| Department | Filename | Rows | Columns |\n")
                f.write("|------------|----------|------|----------|\n")
                for sheet_info in self.spreadsheet_info:
                    if 'error' not in sheet_info:
                        f.write(f"| {sheet_info['department']} | {sheet_info['filename']} | "
                               f"{sheet_info.get('num_rows', 'N/A')} | "
                               f"{sheet_info.get('num_columns', 'N/A')} |\n")
                f.write("\n")
            
            # Footer
            f.write("---\n\n")
            f.write("*Report generated automatically by audio_analysis pipeline*\n")
        
        print(f"  ✓ Markdown report saved to {report_path}")
    
    def generate_report(self):
        """Generate complete report"""
        self.load_data()
        self.create_visualizations()
        self.generate_markdown_report()
        
        print(f"\n{'='*60}")
        print("REPORT GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"All outputs saved to: {self.output_dir}")
        print(f"\nView the report: {self.output_dir / 'analysis_report.md'}")


def main():
    analysis_dir = "/Users/chaitanyakartik/Projects/asr-finetuning/call_recording_analysis"
    
    print("="*60)
    print("Generating Comprehensive Analysis Report")
    print("="*60)
    
    generator = ReportGenerator(analysis_dir)
    generator.generate_report()


if __name__ == "__main__":
    main()
