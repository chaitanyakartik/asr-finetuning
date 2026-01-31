"""
Audio Analysis Script for AI-ML Call Recordings
Analyzes MP3 files from government department call centers
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from pydub import AudioSegment
    from pydub.utils import mediainfo
    AUDIO_LIB = 'pydub'
except ImportError:
    print("pydub not found, trying librosa...")
    try:
        import librosa
        import soundfile as sf
        AUDIO_LIB = 'librosa'
    except ImportError:
        print("WARNING: Neither pydub nor librosa found. Install one of them:")
        print("  pip install pydub")
        print("  or")
        print("  pip install librosa soundfile")
        AUDIO_LIB = None

import pandas as pd
from tqdm import tqdm


class AudioAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.results = []
        self.stats = defaultdict(lambda: defaultdict(list))
        
    def get_audio_info_pydub(self, audio_path):
        """Extract audio information using pydub"""
        try:
            audio = AudioSegment.from_mp3(str(audio_path))
            info = mediainfo(str(audio_path))
            
            return {
                'duration_seconds': len(audio) / 1000.0,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'sample_width': audio.sample_width,
                'bit_rate': info.get('bit_rate', 'N/A'),
                'codec': info.get('codec_name', 'N/A'),
                'file_size_mb': audio_path.stat().st_size / (1024 * 1024),
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def get_audio_info_librosa(self, audio_path):
        """Extract audio information using librosa"""
        try:
            y, sr = librosa.load(str(audio_path), sr=None, mono=False)
            duration = librosa.get_duration(y=y, sr=sr)
            
            if y.ndim == 1:
                channels = 1
            else:
                channels = y.shape[0]
            
            return {
                'duration_seconds': duration,
                'sample_rate': sr,
                'channels': channels,
                'sample_width': 'N/A',
                'bit_rate': 'N/A',
                'codec': 'mp3',
                'file_size_mb': audio_path.stat().st_size / (1024 * 1024),
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def get_audio_info(self, audio_path):
        """Get audio info using available library"""
        if AUDIO_LIB == 'pydub':
            return self.get_audio_info_pydub(audio_path)
        elif AUDIO_LIB == 'librosa':
            return self.get_audio_info_librosa(audio_path)
        else:
            # Fallback to file size only
            return {
                'duration_seconds': 'N/A',
                'sample_rate': 'N/A',
                'channels': 'N/A',
                'sample_width': 'N/A',
                'bit_rate': 'N/A',
                'codec': 'N/A',
                'file_size_mb': audio_path.stat().st_size / (1024 * 1024),
            }
    
    def find_all_audio_files(self):
        """Find all MP3 files in the directory structure"""
        audio_files = list(self.base_path.rglob("*.mp3"))
        print(f"Found {len(audio_files)} MP3 files")
        return audio_files
    
    def analyze_directory_structure(self):
        """Analyze the directory structure and group files by department"""
        audio_files = self.find_all_audio_files()
        
        # Group by department (first level subdirectory)
        departments = defaultdict(list)
        for audio_file in audio_files:
            # Get relative path from base
            rel_path = audio_file.relative_to(self.base_path)
            department = rel_path.parts[0] if len(rel_path.parts) > 0 else "Unknown"
            departments[department].append(audio_file)
        
        return departments
    
    def analyze_all_files(self, limit=None):
        """Analyze all audio files or up to limit"""
        print("Finding audio files...")
        audio_files = self.find_all_audio_files()
        
        if limit:
            audio_files = audio_files[:limit]
            print(f"Analyzing first {limit} files...")
        
        print(f"Analyzing {len(audio_files)} audio files...")
        
        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            info = self.get_audio_info(audio_path)
            
            if info:
                rel_path = audio_path.relative_to(self.base_path)
                department = rel_path.parts[0] if len(rel_path.parts) > 0 else "Unknown"
                
                result = {
                    'file_path': str(audio_path),
                    'relative_path': str(rel_path),
                    'filename': audio_path.name,
                    'department': department,
                    **info
                }
                
                self.results.append(result)
                
                # Collect stats by department
                if info['duration_seconds'] != 'N/A':
                    self.stats[department]['durations'].append(info['duration_seconds'])
                    self.stats[department]['sample_rates'].append(info['sample_rate'])
                    self.stats[department]['file_sizes'].append(info['file_size_mb'])
                    self.stats[department]['channels'].append(info['channels'])
        
        return self.results
    
    def generate_summary_stats(self):
        """Generate summary statistics"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        # Overall stats
        summary = {
            'total_files': len(df),
            'total_departments': df['department'].nunique(),
            'departments': sorted(df['department'].unique().tolist()),
        }
        
        # Audio stats (if available)
        if df['duration_seconds'].dtype != 'object':
            summary.update({
                'total_duration_hours': df['duration_seconds'].sum() / 3600,
                'avg_duration_seconds': df['duration_seconds'].mean(),
                'median_duration_seconds': df['duration_seconds'].median(),
                'min_duration_seconds': df['duration_seconds'].min(),
                'max_duration_seconds': df['duration_seconds'].max(),
                'total_size_gb': df['file_size_mb'].sum() / 1024,
            })
        
        # Sample rates
        if df['sample_rate'].dtype != 'object':
            summary['sample_rates'] = df['sample_rate'].value_counts().to_dict()
            summary['most_common_sample_rate'] = int(df['sample_rate'].mode()[0])
        
        # Channels
        if df['channels'].dtype != 'object':
            summary['channel_distribution'] = df['channels'].value_counts().to_dict()
        
        # Per-department stats
        dept_stats = {}
        for dept in df['department'].unique():
            dept_df = df[df['department'] == dept]
            dept_stat = {
                'file_count': len(dept_df),
                'percentage': (len(dept_df) / len(df)) * 100,
            }
            
            if dept_df['duration_seconds'].dtype != 'object':
                dept_stat.update({
                    'total_duration_hours': dept_df['duration_seconds'].sum() / 3600,
                    'avg_duration_seconds': dept_df['duration_seconds'].mean(),
                    'median_duration_seconds': dept_df['duration_seconds'].median(),
                    'total_size_mb': dept_df['file_size_mb'].sum(),
                })
            
            dept_stats[dept] = dept_stat
        
        summary['department_stats'] = dept_stats
        
        return summary
    
    def save_results(self, output_dir):
        """Save analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        df = pd.DataFrame(self.results)
        df.to_csv(output_dir / 'detailed_audio_analysis.csv', index=False)
        print(f"Saved detailed analysis to {output_dir / 'detailed_audio_analysis.csv'}")
        
        # Save summary stats
        summary = self.generate_summary_stats()
        with open(output_dir / 'summary_stats.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved summary stats to {output_dir / 'summary_stats.json'}")
        
        # Save department summary as CSV
        if 'department_stats' in summary:
            dept_df = pd.DataFrame(summary['department_stats']).T
            dept_df.index.name = 'department'
            dept_df.to_csv(output_dir / 'department_summary.csv')
            print(f"Saved department summary to {output_dir / 'department_summary.csv'}")
        
        return summary


def main():
    base_path = "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML"
    output_dir = "/Users/chaitanyakartik/Projects/asr-finetuning/call_recording_analysis"
    
    print("="*60)
    print("Call Recording Audio Analysis")
    print("="*60)
    
    analyzer = AudioAnalyzer(base_path)
    
    # Analyze all files (or set limit for testing)
    # analyzer.analyze_all_files(limit=50)  # Test with first 50 files
    analyzer.analyze_all_files()  # Analyze all files
    
    # Save results
    summary = analyzer.save_results(output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Files: {summary['total_files']}")
    print(f"Total Departments: {summary['total_departments']}")
    
    if 'total_duration_hours' in summary:
        print(f"Total Duration: {summary['total_duration_hours']:.2f} hours")
        print(f"Average Duration: {summary['avg_duration_seconds']:.2f} seconds")
        print(f"Median Duration: {summary['median_duration_seconds']:.2f} seconds")
        print(f"Total Size: {summary['total_size_gb']:.2f} GB")
    
    if 'most_common_sample_rate' in summary:
        print(f"Most Common Sample Rate: {summary['most_common_sample_rate']} Hz")
    
    print("\nTop 5 Departments by File Count:")
    dept_stats = summary.get('department_stats', {})
    sorted_depts = sorted(dept_stats.items(), key=lambda x: x[1]['file_count'], reverse=True)
    for dept, stats in sorted_depts[:5]:
        print(f"  {dept}: {stats['file_count']} files ({stats['percentage']:.1f}%)")


if __name__ == "__main__":
    main()
