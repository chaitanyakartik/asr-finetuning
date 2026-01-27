# Call Recording Analysis

This directory contains scripts for analyzing call recordings from various government departments.

## Overview

- **816 MP3 audio files** across **27 departments**
- Each department has a subfolder with audio files and an Excel spreadsheet with metadata
- Audio files are nested 3 levels deep in the directory structure

## Scripts

### 1. `audio_analysis.py`
Analyzes all audio files and extracts:
- Duration statistics
- Sample rates
- Channel information (mono/stereo)
- File sizes
- Per-department breakdown

**Output:**
- `detailed_audio_analysis.csv` - Complete audio file information
- `summary_stats.json` - Overall statistics
- `department_summary.csv` - Per-department summary

### 2. `spreadsheet_analysis.py`
Analyzes Excel spreadsheets associated with recordings:
- Extracts column information
- Identifies audio-related fields
- Correlates with audio files
- Exports spreadsheets to CSV format

**Output:**
- `spreadsheet_analysis.json` - Spreadsheet metadata
- `spreadsheets_csv/` - All spreadsheets converted to CSV
- `audio_spreadsheet_correlation.json` - Correlation between audio and metadata

### 3. `generate_report.py`
Creates comprehensive analysis report with visualizations:
- Distribution plots (duration, sample rate, file size)
- Per-department comparisons
- Quality assessment for ASR training
- Recommendations

**Output:**
- `reports/analysis_report.md` - Comprehensive markdown report
- `reports/*.png` - Visualization charts

### 4. `run_full_analysis.py`
Master script that runs all analysis steps in sequence.

## Installation

Install required dependencies:

```bash
# Option 1: Using pydub (recommended for MP3)
pip install pandas openpyxl pydub tqdm matplotlib seaborn

# For pydub to work with MP3, you also need ffmpeg:
# macOS:
brew install ffmpeg

# Option 2: Using librosa (alternative)
pip install pandas openpyxl librosa soundfile tqdm matplotlib seaborn
```

## Usage

### Quick Start - Run Everything

```bash
cd /Users/chaitanyakartik/Projects/asr-finetuning/call_recording_analysis
python run_full_analysis.py
```

### Run Individual Scripts

```bash
# 1. Analyze audio files
python audio_analysis.py

# 2. Analyze spreadsheets
python spreadsheet_analysis.py

# 3. Generate report
python generate_report.py
```

### Test on Subset

To test on a limited number of files first, edit `audio_analysis.py`:

```python
# In main() function, change:
analyzer.analyze_all_files()
# To:
analyzer.analyze_all_files(limit=50)  # Analyze first 50 files
```

## Output Structure

```
call_recording_analysis/
├── README.md
├── audio_analysis.py
├── spreadsheet_analysis.py
├── generate_report.py
├── run_full_analysis.py
├── detailed_audio_analysis.csv
├── summary_stats.json
├── department_summary.csv
├── spreadsheet_analysis.json
├── audio_spreadsheet_correlation.json
├── spreadsheets_csv/
│   └── [department_spreadsheets].csv
└── reports/
    ├── analysis_report.md
    ├── files_per_department.png
    ├── duration_distribution.png
    ├── sample_rate_distribution.png
    ├── file_size_distribution.png
    ├── department_comparison.png
    └── duration_by_department_boxplot.png
```

## Context: ASR Training & Testing

This analysis is designed to prepare data for ASR (Automatic Speech Recognition) training and testing:

### Key Metrics for ASR
- **Sample Rate**: Should be consistent (typically 16kHz for ASR)
- **Duration**: Optimal range 3-30 seconds per clip
- **Format**: Mono audio preferred
- **Quality**: Signal-to-noise ratio, background noise assessment

### Next Steps After Analysis
1. **Transcription**: Generate transcriptions using Whisper or manual annotation
2. **Preprocessing**: Normalize, resample, convert to mono
3. **Dataset Creation**: Create train/val/test splits
4. **Manifest Files**: Generate NeMo-compatible manifest files

## Troubleshooting

### No audio library found
Install pydub and ffmpeg:
```bash
pip install pydub
brew install ffmpeg  # macOS
```

Or install librosa:
```bash
pip install librosa soundfile
```

### Excel reading errors
Make sure openpyxl is installed:
```bash
pip install openpyxl
```

### Memory issues with large files
Process files in batches by modifying the limit parameter.

## Data Structure

```
AI-ML/
├── Agriculture Department/
│   ├── Department of Agriculture/
│   │   └── Department of Agriculture/
│   │       ├── 215854-.mp3
│   │       ├── 216145-.mp3
│   │       └── ...
│   └── Untitled spreadsheet.xlsx
├── Health and Family Welfare Department/
│   ├── Health & Family Welfare Services/
│   │   └── ...
│   └── Untitled spreadsheet.xlsx
└── [25 more departments...]
```

## Performance Notes

- Full analysis of 816 files takes approximately 5-15 minutes depending on system
- Audio processing is the most time-consuming step
- Visualizations are generated quickly once data is processed
- Results are cached, so re-running report generation is fast

## Support

For issues or questions, check the script output for detailed error messages.
