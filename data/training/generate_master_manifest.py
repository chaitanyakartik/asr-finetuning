#!/usr/bin/env python3
"""
Generate master manifest from all subfolders in a version directory
Usage: python generate_master_manifest.py v2
"""

import os
import sys
import json
import re
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

def clean_text(text):
    """Clean text by removing tags, English translations, and formatting symbols"""
    # Remove only the tags themselves, keep the content inside
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove English words wrapped in {} like {chair}, {picture}, etc.
    text = re.sub(r'\s*\{[^}]+\}', '', text)
    
    # Remove formatting symbols: - _ @ # $ ! % & *
    text = re.sub(r'[-_@#$!%&*]', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def generate_master_manifest(version):
    """Generate master manifest for a specific version"""
    print(f"--- 🔗 Generating Master Manifest for {version} ---")
    
    version_dir = BASE_DIR / version
    
    if not version_dir.exists():
        print(f"❌ Error: Version directory not found: {version_dir}")
        return
    
    # Find all subfolders
    subfolders = [d for d in version_dir.iterdir() if d.is_dir()]
    
    if not subfolders:
        print(f"❌ Error: No subfolders found in {version_dir}")
        return
    
    print(f"   📂 Found {len(subfolders)} datasets")
    
    master_entries = []
    total_duration = 0.0
    
    for subfolder in sorted(subfolders):
        manifest_path = subfolder / "train_manifest.json"
        
        if not manifest_path.exists():
            print(f"   ⚠️  Skipping {subfolder.name} (no train_manifest.json)")
            continue
        
        print(f"   📖 Reading {subfolder.name}...")
        
        dataset_duration = 0.0
        count = 0
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Clean the text field
                    entry['text'] = clean_text(entry['text'])
                    master_entries.append(entry)
                    dataset_duration += entry.get('duration', 0)
                    count += 1
        
        print(f"      + Added {count} items ({dataset_duration/3600:.2f} hours)")
        total_duration += dataset_duration
    
    # Write Master Manifest
    output_path = version_dir / "master_manifest.json"
    print(f"\n   💾 Saving to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in master_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    
    print("-" * 60)
    print(f"✅ MASTER MANIFEST COMPLETE")
    print(f"   VERSION: {version}")
    print(f"   TOTAL SAMPLES: {len(master_entries)}")
    print(f"   TOTAL DURATION: {total_duration/3600:.2f} hours")
    print(f"   LOCATION: {output_path.absolute()}")
    print("-" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_master_manifest.py <version>")
        print("Example: python generate_master_manifest.py v2")
        sys.exit(1)
    
    version = sys.argv[1]
    generate_master_manifest(version)
