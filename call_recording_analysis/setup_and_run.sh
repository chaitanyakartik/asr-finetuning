#!/bin/bash

# Setup and run call recording analysis
# This script activates the venv and installs dependencies

echo "=========================================="
echo "Call Recording Analysis Setup"
echo "=========================================="

# Navigate to project root
cd /Users/chaitanyakartik/Projects/asr-finetuning

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing required packages..."
pip install -q pandas openpyxl tqdm matplotlib seaborn pydub

# Check if ffmpeg is installed (needed for pydub to work with MP3)
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "⚠️  WARNING: ffmpeg not found!"
    echo "   Audio analysis will be limited without it."
    echo "   Install with: brew install ffmpeg"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Navigate to analysis directory
cd call_recording_analysis

echo ""
echo "=========================================="
echo "Running Analysis Pipeline"
echo "=========================================="
echo ""

# Run the full analysis
python run_full_analysis.py

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "View the report at:"
echo "  call_recording_analysis/reports/analysis_report.md"
echo ""
