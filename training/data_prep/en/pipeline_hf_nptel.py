import json
import os
from datasets import load_dataset

# --- CONFIG ---
OUTPUT_DIR = "data/training/v3/en/nptel"
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "train_manifest.json")
METADATA_PATH = os.path.join(OUTPUT_DIR, "raw_metadata.json")
NEW_MANIFEST_PATH = os.path.join(OUTPUT_DIR, "train_manifest_fixed.json")

HF_DATASET_ID = "skbose/indian-english-nptel-v0"
CACHE_DIR = "/mnt/data/hf_cache"

def repair_manifest():
    print("🛠️ Starting Manifest Repair...")

    # 1. Load the HF dataset (streaming=True is faster if we just need text)
    print(f"📡 Connecting to {HF_DATASET_ID} to fetch normalized text...")
    ds = load_dataset(HF_DATASET_ID, split="train", cache_dir=CACHE_DIR)

    # 2. Load your local raw_metadata to get the mapping
    with open(METADATA_PATH, "r") as f:
        local_metadata = json.load(f)

    # 3. Load the current (incorrect) manifest entries
    with open(MANIFEST_PATH, "r") as f:
        manifest_lines = f.readlines()

    if len(local_metadata) != len(manifest_lines):
        print("❌ Error: Metadata and Manifest row counts don't match!")
        return

    fixed_entries = []

    print(f"📝 Updating {len(manifest_lines)} entries...")
    for i, line in enumerate(manifest_lines):
        entry = json.loads(line)
        
        # Get the original index we stored during the first run
        original_idx = local_metadata[i]["original_index"]
        
        # Fetch the normalized text from the dataset
        # Note: dataset[idx] is fast for cached datasets
        normalized_text = ds[original_idx]["transcription_normalised"]
        
        # Update the entry
        entry["text"] = normalized_text
        fixed_entries.append(entry)

    # 4. Save the new manifest
    with open(NEW_MANIFEST_PATH, "w", encoding="utf-8") as f:
        for entry in fixed_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"✅ Success! Fixed manifest saved to: {NEW_MANIFEST_PATH}")
    print(f"Sample Text: {fixed_entries[0]['text']}")

if __name__ == "__main__":
    repair_manifest()