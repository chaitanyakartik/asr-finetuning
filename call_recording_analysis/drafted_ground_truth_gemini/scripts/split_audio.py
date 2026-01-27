#!/usr/bin/env python3
"""
Split audio file into 30-second chunks
"""

from pydub import AudioSegment
import os

# Input and output paths
input_file = "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/Housing Department/housing department/housing department/228016.mp3"
output_dir = "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/test_data"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load audio file
print(f"Loading audio file: {input_file}")
audio = AudioSegment.from_file(input_file)

# Get duration in milliseconds
duration_ms = len(audio)
duration_sec = duration_ms / 1000
print(f"Audio duration: {duration_sec:.2f} seconds")

# Split into 30-second chunks
chunk_length_ms = 30 * 1000  # 30 seconds in milliseconds
chunks = []

for i in range(0, duration_ms, chunk_length_ms):
    chunk = audio[i:i + chunk_length_ms]
    chunk_number = (i // chunk_length_ms) + 1
    output_file = os.path.join(output_dir, f"228016_part{chunk_number:02d}.mp3")
    
    print(f"Exporting chunk {chunk_number}: {output_file} ({len(chunk)/1000:.2f}s)")
    chunk.export(output_file, format="mp3")
    chunks.append(output_file)

print(f"\n✅ Successfully split audio into {len(chunks)} chunks")
print(f"Output directory: {output_dir}")
