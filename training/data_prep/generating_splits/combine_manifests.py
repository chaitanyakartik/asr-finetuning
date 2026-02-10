import json
import os

# --- CONFIGURATION ---
# List all manifest paths you want to combine
INPUT_MANIFESTS = [
    "/mnt/data/asr-finetuning/data/training/v3/hi/indicvoices/train_manifest.json",
    "/mnt/data/asr-finetuning/data/training/v3/hi/kathbath/train_manifest.json",
    "/mnt/data/asr-finetuning/data/training/v3/hi/shrutilipi/train_manifest.json",
    "/mnt/data/asr-finetuning/data/training/v3/hi/vaani/train_manifest.json",
]
OUTPUT_PATH = "/mnt/data/asr-finetuning/data/training/v3/hi/train_manifest_combined.json"

def combine_manifests():
    print(f"--- 🌀 Combining {len(INPUT_MANIFESTS)} manifests ---")
    
    total_lines = 0
    total_duration = 0
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as outfile:
        for manifest_path in INPUT_MANIFESTS:
            if not os.path.exists(manifest_path):
                print(f"⚠️ Skipping: {manifest_path} (File not found)")
                continue
                
            print(f"📂 Processing: {os.path.basename(manifest_path)}")
            
            with open(manifest_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if line.strip():
                        # Optional: Calculate stats while writing
                        entry = json.loads(line)
                        total_duration += entry.get('duration', 0)
                        
                        # Write directly to avoid loading everything into memory
                        outfile.write(line.strip() + '\n')
                        total_lines += 1

    print(f"\n✅ Combination Complete!")
    print(f"📊 Total Entries: {total_lines}")
    print(f"⏱️ Total Duration: {total_duration / 3600:.2f} hours")
    print(f"💾 Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    combine_manifests()