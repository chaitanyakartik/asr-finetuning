#!/usr/bin/env python3
"""
Regenerate train_manifest.json files with absolute paths for all datasets in a directory.

This script walks through a base directory (e.g., v2/), finds all subdirectories with:
- wavs/ folder
- raw_metadata.json

And regenerates train_manifest.json with correct absolute file paths.
"""

import os
import json
import argparse
from pathlib import Path
from pydub import AudioSegment


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds"""
    try:
        audio = AudioSegment.from_file(audio_path)
        return round(len(audio) / 1000.0, 7)
    except Exception as e:
        print(f"      ⚠️  Error getting duration for {audio_path}: {e}")
        return 0.0


def read_manifest_for_text(manifest_path):
    """Read existing manifest to get text content"""
    text_map = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        filename = os.path.basename(entry['audio_filepath'])
                        text_map[filename] = entry.get('text', '')
        except Exception as e:
            print(f"      ⚠️  Error reading existing manifest: {e}")
    return text_map


def regenerate_manifest(dataset_dir):
    """Regenerate train_manifest.json for a single dataset directory"""
    dataset_name = os.path.basename(dataset_dir)
    print(f"\n📁 Processing: {dataset_name}")
    
    # Check for required files/directories
    wavs_dir = os.path.join(dataset_dir, "wavs")
    raw_metadata_path = os.path.join(dataset_dir, "raw_metadata.json")
    old_manifest_path = os.path.join(dataset_dir, "train_manifest.json")
    
    if not os.path.isdir(wavs_dir):
        print(f"   ⚠️  No wavs/ directory found, skipping")
        return False
    
    if not os.path.exists(raw_metadata_path):
        print(f"   ⚠️  No raw_metadata.json found, skipping")
        return False
    
    # Load raw metadata
    with open(raw_metadata_path, 'r', encoding='utf-8') as f:
        raw_metadata = json.load(f)
    
    print(f"   Found {len(raw_metadata)} entries in raw_metadata.json")
    
    # Read existing manifest to get text content
    text_map = read_manifest_for_text(old_manifest_path)
    
    # Generate new manifest entries
    manifest_entries = []
    wav_files = set(os.listdir(wavs_dir))
    
    for metadata in raw_metadata:
        filename = metadata.get('filename')
        if not filename:
            continue
        
        # Check if WAV file exists
        wav_path = os.path.join(wavs_dir, filename)
        if filename not in wav_files:
            print(f"   ⚠️  WAV file not found: {filename}")
            continue
        
        # Get text from old manifest or use empty string
        text = text_map.get(filename, "")
        
        # Get duration
        duration = get_audio_duration(wav_path)
        
        # Determine source
        source = metadata.get('category', dataset_name.lower())
        if 'source' in metadata:
            source = metadata['source']
        
        # Create manifest entry
        entry = {
            "audio_filepath": wav_path,  # Absolute path
            "text": text,
            "duration": duration,
            "lang": "kn",
            "source": source
        }
        
        manifest_entries.append(entry)
    
    # Write new manifest
    new_manifest_path = os.path.join(dataset_dir, "train_manifest.json")
    with open(new_manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Generated train_manifest.json with {len(manifest_entries)} entries")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate train_manifest.json files with absolute paths"
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing dataset subdirectories (e.g., /path/to/v2)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes"
    )
    
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.base_dir)
    
    if not os.path.isdir(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return 1
    
    print("=" * 80)
    print(f"Regenerating manifests in: {base_dir}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)
    
    # Find all subdirectories
    processed = 0
    skipped = 0
    
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        
        if not os.path.isdir(item_path):
            continue
        
        # Skip hidden directories
        if item.startswith('.'):
            continue
        
        if args.dry_run:
            # Check if it has wavs/ and raw_metadata.json
            has_wavs = os.path.isdir(os.path.join(item_path, "wavs"))
            has_metadata = os.path.exists(os.path.join(item_path, "raw_metadata.json"))
            
            if has_wavs and has_metadata:
                print(f"\n📁 Would process: {item}")
                processed += 1
            else:
                print(f"\n⏭️  Would skip: {item} (missing wavs/ or raw_metadata.json)")
                skipped += 1
        else:
            # Actually process
            if regenerate_manifest(item_path):
                processed += 1
            else:
                skipped += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print("=" * 80)
    
    if args.dry_run:
        print("\n💡 Run without --dry-run to actually generate manifests")
    else:
        print("\n✅ All manifests regenerated!")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
