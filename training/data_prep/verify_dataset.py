#!/usr/bin/env python3
"""
Verify dataset completeness by comparing manifest entries with audio files
Usage: python verify_dataset.py v2
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

def verify_dataset(dataset_path):
    """Verify a single dataset"""
    manifest = dataset_path / "train_manifest.json"
    
    # Handle different directory structures
    dataset_name = dataset_path.name.lower()
    if dataset_name == "iisc_mile":
        wavs_dir = dataset_path / "raw_extract" / "train" / "audio_files"
    else:
        wavs_dir = dataset_path / "wavs"
    
    if not manifest.exists():
        print(f"   ⚠️  No manifest found")
        return None
    
    if not wavs_dir.exists():
        print(f"   ⚠️  No audio directory found at {wavs_dir.relative_to(dataset_path)}")
        return None
    
    # Read manifest and collect durations
    durations = []
    manifest_count = 0
    
    with open(manifest) as f:
        for line in f:
            if line.strip():
                manifest_count += 1
                try:
                    entry = json.loads(line)
                    durations.append(entry.get('duration', 0))
                except:
                    pass
    
    # Count wav files
    wav_count = len(list(wavs_dir.glob("*.wav")))
    
    total_hours = sum(durations) / 3600
    
    print(f"   Manifest: {manifest_count:,} | Audio: {wav_count:,} | Hours: {total_hours:.2f}", end=" ")
    
    if manifest_count == wav_count:
        print("✅")
        return {"complete": True, "count": manifest_count, "hours": total_hours, "durations": durations}
    else:
        print(f"❌ (diff: {abs(manifest_count - wav_count)})")
        return {"complete": False, "count": manifest_count, "hours": total_hours, "durations": durations}

def show_duration_distribution(all_durations):
    """Show duration distribution as a histogram"""
    if not all_durations:
        return
    
    # Define buckets (in seconds)
    buckets = [
        (0, 1.5, "0-1.5s"),
	(1.5,2, "1.5-2s"),
        (2, 5, "2-5s"),
        (5, 10, "5-10s"),
        (10, 15, "10-15s"),
        (15, 20, "15-20s"),
        (20, float('inf'), "20s+")
    ]
    
    bucket_counts = defaultdict(int)
    
    for duration in all_durations:
        for min_d, max_d, label in buckets:
            if min_d <= duration < max_d:
                bucket_counts[label] += 1
                break
    
    total = len(all_durations)
    
    print("\n📊 Duration Distribution:")
    print("-" * 50)
    
    max_count = max(bucket_counts.values()) if bucket_counts else 1
    
    for _, _, label in buckets:
        count = bucket_counts[label]
        percentage = (count / total * 100) if total > 0 else 0
        bar_length = int((count / max_count) * 30) if max_count > 0 else 0
        bar = "█" * bar_length
        print(f"  {label:8} | {bar:30} {count:8,} ({percentage:5.1f}%)")

def verify_version(version):
    """Verify all datasets in a version directory"""
    base_path = Path("data/training") / version
    
    if not base_path.exists():
        print(f"❌ Version directory not found: {base_path}")
        return
    
    # Find all subdirectories
    datasets = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    if not datasets:
        print(f"❌ No datasets found in {base_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Verifying {version} datasets ({len(datasets)} total)")
    print(f"{'='*60}\n")
    
    complete = 0
    incomplete = 0
    total_hours = 0
    total_samples = 0
    all_durations = []
    
    for dataset in datasets:
        print(f"📊 {dataset.name}")
        result = verify_dataset(dataset)
        if result:
            if result["complete"]:
                complete += 1
            else:
                incomplete += 1
            total_hours += result["hours"]
            total_samples += result["count"]
            all_durations.extend(result["durations"])
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Datasets: {complete} complete, {incomplete} incomplete")
    print(f"  Total Samples: {total_samples:,}")
    print(f"  Total Hours: {total_hours:.2f} ({total_hours/3600:.2f} days)" if total_hours > 100 else f"  Total Hours: {total_hours:.2f}")
    print(f"{'='*60}")
    
    show_duration_distribution(all_durations)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_dataset.py <version>")
        print("Example: python verify_dataset.py v2")
        sys.exit(1)
    
    verify_version(sys.argv[1])
