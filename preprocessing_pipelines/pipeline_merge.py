import os
import json
from tqdm import tqdm

# --- CONFIGURATION ---
PROCESSED_ROOT = "processed_data"
OUTPUT_DIR = "final_dataset"
OUTPUT_MANIFEST = os.path.join(OUTPUT_DIR, "master_train_manifest.json")

# The subfolders to merge
DATASETS = ["Shrutilipi", "Kathbath", "IndicVoices", "Vaani"]

def merge_manifests():
    print("--- 🔗 Starting Manifest Merge ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    master_entries = []
    total_duration = 0.0
    
    for ds_name in DATASETS:
        manifest_path = os.path.join(PROCESSED_ROOT, ds_name, "train_manifest.json")
        
        if not os.path.exists(manifest_path):
            print(f"⚠️ Warning: Manifest not found for {ds_name} at {manifest_path}")
            continue
            
        print(f"   📖 Reading {ds_name}...")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        dataset_duration = 0.0
        for line in lines:
            entry = json.loads(line)
            master_entries.append(entry)
            dataset_duration += entry['duration']
            
        print(f"      + Added {len(lines)} items ({dataset_duration/3600:.2f} hours)")
        total_duration += dataset_duration

    # Write Master File
    print(f"\n   💾 Saving Master Manifest to {OUTPUT_MANIFEST}...")
    with open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f:
        for entry in master_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print("-" * 40)
    print(f"✅ MERGE COMPLETE")
    print(f"   TOTAL SAMPLES: {len(master_entries)}")
    print(f"   TOTAL DURATION: {total_duration/3600:.4f} hours")
    print(f"   LOCATION: {os.path.abspath(OUTPUT_MANIFEST)}")
    print("-" * 40)

if __name__ == "__main__":
    merge_manifests()