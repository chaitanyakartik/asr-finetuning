#!/usr/bin/env python3
"""
Create manifest file from transcribed audio chunks
"""
import os
import json
from pydub import AudioSegment

# Paths
AUDIO_DIR = "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/test_data/228016"
TRANSCRIPTIONS_FILE = os.path.join(AUDIO_DIR, "transcriptions_gemini_2.5_flash.json")
OUTPUT_MANIFEST = "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/test_data/manifest.json"

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds"""
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # Convert ms to seconds
    except Exception as e:
        print(f"Error getting duration for {audio_path}: {e}")
        return 0.0

def main():
    print(f"📝 Creating manifest file...")
    print(f"Reading transcriptions from: {TRANSCRIPTIONS_FILE}")
    
    # Load transcriptions
    with open(TRANSCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
        transcriptions = json.load(f)
    
    # Create manifest entries
    manifest_entries = []
    
    for item in transcriptions:
        if "transcription" not in item:
            print(f"⚠️  Skipping {item['filename']} - no transcription")
            continue
        
        audio_path = os.path.join(AUDIO_DIR, item['filename'])
        
        # Get duration
        duration = get_audio_duration(audio_path)
        
        # Create manifest entry
        entry = {
            "audio_filepath": audio_path,
            "text": item['transcription'],
            "duration": round(duration, 7),
            "lang": "kn",
            "source": "housing_dept_call"
        }
        
        manifest_entries.append(entry)
        print(f"✅ {item['filename']}: {duration:.2f}s")
    
    # Write manifest file (one JSON object per line)
    with open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Manifest created: {OUTPUT_MANIFEST}")
    print(f"Total entries: {len(manifest_entries)}")

if __name__ == "__main__":
    main()
