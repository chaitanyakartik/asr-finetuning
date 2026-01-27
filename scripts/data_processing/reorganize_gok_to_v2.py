#!/usr/bin/env python3
"""
Reorganize GoK call recordings to match v2 dataset format
"""
import os
import json
import shutil
from pathlib import Path
from pydub import AudioSegment

# Paths
BASE_DIR = "/Users/chaitanyakartik/Projects/asr-finetuning/evaluation/benchmarking/data/v2/gok_call_recordings"
SOURCE_DIR = "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/drafted_ground_truth_gemini"
MASTER_MANIFEST = os.path.join(SOURCE_DIR, "manifests/manifest_master.json")

def create_directory_structure():
    """Create v2 format directory structure"""
    print("📁 Creating directory structure...")
    os.makedirs(os.path.join(BASE_DIR, "wavs"), exist_ok=True)
    print("   ✅ Created wavs/")

def convert_and_copy_audio():
    """Convert MP3 chunks to WAV and copy to wavs/"""
    print("\n🔄 Converting and copying audio files...")
    
    audio_chunks_dir = os.path.join(SOURCE_DIR, "audio_chunks")
    wavs_dir = os.path.join(BASE_DIR, "wavs")
    
    converted_files = []
    
    for dept_dir in os.listdir(audio_chunks_dir):
        dept_path = os.path.join(audio_chunks_dir, dept_dir)
        if not os.path.isdir(dept_path):
            continue
        
        print(f"   Processing {dept_dir}...")
        
        for mp3_file in sorted(os.listdir(dept_path)):
            if not mp3_file.endswith('.mp3'):
                continue
            
            mp3_path = os.path.join(dept_path, mp3_file)
            
            # Create WAV filename: dept_partXX.wav
            wav_filename = f"{dept_dir}_{mp3_file.replace('.mp3', '.wav')}"
            wav_path = os.path.join(wavs_dir, wav_filename)
            
            # Convert MP3 to WAV
            audio = AudioSegment.from_file(mp3_path)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            audio.export(wav_path, format='wav')
            
            converted_files.append({
                "original": mp3_path,
                "wav": wav_path,
                "filename": wav_filename,
                "dept": dept_dir
            })
    
    print(f"   ✅ Converted {len(converted_files)} files to WAV")
    return converted_files

def create_train_manifest(converted_files):
    """Create train_manifest.json in v2 format"""
    print("\n📝 Creating train_manifest.json...")
    
    # Load original master manifest
    with open(MASTER_MANIFEST, 'r', encoding='utf-8') as f:
        original_entries = [json.loads(line) for line in f]
    
    # Create mapping from original filename to entry
    filename_to_entry = {}
    for entry in original_entries:
        filename = os.path.basename(entry['audio_filepath'])
        filename_to_entry[filename] = entry
    
    # Create train manifest entries
    manifest_entries = []
    
    for file_info in converted_files:
        original_filename = os.path.basename(file_info['original'])
        original_entry = filename_to_entry.get(original_filename)
        
        if not original_entry:
            print(f"   ⚠️  No entry found for {original_filename}")
            continue
        
        # Create entry matching v2 format
        entry = {
            "audio_filepath": file_info['wav'],
            "text": original_entry['text'],
            "duration": original_entry['duration'],
            "lang": "kn",
            "source": "gok_call_recordings"
        }
        
        manifest_entries.append(entry)
    
    # Write train_manifest.json
    manifest_path = os.path.join(BASE_DIR, "train_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Created train_manifest.json with {len(manifest_entries)} entries")
    return manifest_entries

def create_raw_metadata(converted_files, manifest_entries):
    """Create raw_metadata.json in v2 format"""
    print("\n📝 Creating raw_metadata.json...")
    
    metadata = []
    
    for idx, (file_info, manifest_entry) in enumerate(zip(converted_files, manifest_entries)):
        entry = {
            "original_index": idx,
            "filename": file_info['filename'],
            "sr": 16000,
            "gender": "Unknown",
            "district": "Unknown",
            "category": "call_recording",
            "scenario": "Government Service Call",
            "task_name": "Call Recording",
            "department": file_info['dept']
        }
        metadata.append(entry)
    
    # Write raw_metadata.json
    metadata_path = os.path.join(BASE_DIR, "raw_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"   ✅ Created raw_metadata.json with {len(metadata)} entries")

def cleanup_old_structure():
    """Remove old directory structure"""
    print("\n🗑️  Cleaning up old structure...")
    
    for dir_name in ['audio_chunks', 'manifests', 'transcriptions']:
        dir_path = os.path.join(BASE_DIR, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"   ✅ Removed {dir_name}/")

def main():
    print("🚀 Reorganizing GoK call recordings to v2 format")
    print("="*80)
    
    # Step 1: Create directory structure
    create_directory_structure()
    
    # Step 2: Convert and copy audio files
    converted_files = convert_and_copy_audio()
    
    # Step 3: Create train_manifest.json
    manifest_entries = create_train_manifest(converted_files)
    
    # Step 4: Create raw_metadata.json
    create_raw_metadata(converted_files, manifest_entries)
    
    # Step 5: Cleanup
    cleanup_old_structure()
    
    print("\n" + "="*80)
    print("✅ Reorganization complete!")
    print(f"Structure: {BASE_DIR}")
    print("   ├── wavs/")
    print("   ├── train_manifest.json")
    print("   └── raw_metadata.json")

if __name__ == "__main__":
    main()
