#!/usr/bin/env python3
"""
Convert MP3 files to WAV and create v2 format files for gok_call_recordings
"""
import os
import json
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

# Paths
BASE_DIR = "/Users/chaitanyakartik/Projects/asr-finetuning/evaluation/benchmarking/data/v2/gok_call_recordings"
AUDIO_CHUNKS_DIR = os.path.join(BASE_DIR, "audio_chunks")
WAVS_DIR = os.path.join(BASE_DIR, "wavs")
MASTER_MANIFEST = os.path.join(BASE_DIR, "manifests/manifest_master.json")

def convert_mp3_to_wav():
    """Convert all MP3 files to WAV format"""
    print("🔄 Converting MP3 to WAV...")
    
    os.makedirs(WAVS_DIR, exist_ok=True)
    
    # Find all MP3 files
    mp3_files = []
    for dept_dir in os.listdir(AUDIO_CHUNKS_DIR):
        dept_path = os.path.join(AUDIO_CHUNKS_DIR, dept_dir)
        if not os.path.isdir(dept_path):
            continue
        
        for mp3_file in os.listdir(dept_path):
            if mp3_file.endswith('.mp3'):
                mp3_files.append((dept_dir, os.path.join(dept_path, mp3_file)))
    
    print(f"   Found {len(mp3_files)} MP3 files")
    
    file_mapping = []
    
    for dept_name, mp3_path in tqdm(mp3_files, desc="Converting"):
        mp3_filename = Path(mp3_path).stem
        wav_filename = f"{dept_name}_{mp3_filename}.wav"
        wav_path = os.path.join(WAVS_DIR, wav_filename)
        
        # Convert to WAV
        audio = AudioSegment.from_file(mp3_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(wav_path, format='wav')
        
        file_mapping.append({
            "mp3_path": mp3_path,
            "wav_path": wav_path,
            "wav_filename": wav_filename,
            "dept": dept_name
        })
    
    print(f"   ✅ Converted {len(file_mapping)} files")
    return file_mapping

def create_train_manifest(file_mapping):
    """Create train_manifest.json"""
    print("\n📝 Creating train_manifest.json...")
    
    # Load transcriptions from each department
    transcriptions_dir = os.path.join(BASE_DIR, "transcriptions")
    
    all_transcriptions = []
    for trans_file in os.listdir(transcriptions_dir):
        if trans_file.endswith('_transcriptions.json'):
            with open(os.path.join(transcriptions_dir, trans_file), 'r', encoding='utf-8') as f:
                trans_data = json.load(f)
                all_transcriptions.extend(trans_data)
    
    print(f"   Loaded {len(all_transcriptions)} transcription entries")
    
    # Create mapping from filename to transcription
    filename_to_trans = {}
    for trans in all_transcriptions:
        filename_to_trans[trans['filename']] = trans
    
    # Create train manifest
    train_entries = []
    
    for file_info in file_mapping:
        mp3_filename = os.path.basename(file_info['mp3_path'])
        trans_entry = filename_to_trans.get(mp3_filename)
        
        if not trans_entry or 'transcription' not in trans_entry:
            print(f"   ⚠️  No transcription for {mp3_filename}")
            continue
        
        # Get duration from WAV file
        try:
            audio = AudioSegment.from_file(file_info['wav_path'])
            duration = len(audio) / 1000.0
        except:
            duration = 30.0  # Default
        
        entry = {
            "audio_filepath": file_info['wav_path'],
            "text": trans_entry['transcription'],
            "duration": round(duration, 7),
            "lang": "kn",
            "source": "gok_call_recordings"
        }
        train_entries.append(entry)
    
    # Write train_manifest.json
    train_manifest_path = os.path.join(BASE_DIR, "train_manifest.json")
    with open(train_manifest_path, 'w', encoding='utf-8') as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Created with {len(train_entries)} entries")
    return train_entries

def create_raw_metadata(file_mapping):
    """Create raw_metadata.json"""
    print("\n📝 Creating raw_metadata.json...")
    
    # Department mapping
    dept_names = {
        "214601": "Labour Department",
        "216172-": "Minorites Welfare Department", 
        "216233": "Public Works Department",
        "228016": "Housing Department",
        "228368RDPR": "RDPR",
        "230049-": "Medical Education Department"
    }
    
    metadata = []
    
    for idx, file_info in enumerate(file_mapping):
        entry = {
            "original_index": idx,
            "filename": file_info['wav_filename'],
            "sr": 16000,
            "district": "Various",
            "gender": "Unknown",
            "department": dept_names.get(file_info['dept'], file_info['dept']),
            "category": "call_recording",
            "scenario": "Government Service Call"
        }
        metadata.append(entry)
    
    # Write raw_metadata.json
    metadata_path = os.path.join(BASE_DIR, "raw_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"   ✅ Created with {len(metadata)} entries")

def main():
    print("🚀 Converting GoK call recordings to v2 format")
    print("="*80)
    
    # Step 1: Convert MP3 to WAV
    file_mapping = convert_mp3_to_wav()
    
    # Step 2: Create train_manifest.json
    create_train_manifest(file_mapping)
    
    # Step 3: Create raw_metadata.json
    create_raw_metadata(file_mapping)
    
    print("\n" + "="*80)
    print("✅ Conversion complete!")
    print(f"   wavs/: {len(file_mapping)} files")
    print(f"   train_manifest.json")
    print(f"   raw_metadata.json")

if __name__ == "__main__":
    main()
