import json
import random
import os

# --- PATH CONFIGURATION ---
KANNADA_MANIFEST = "/mnt/data/asr-finetuning/data/training/v2.1/master_manifest.json"
ENGLISH_MANIFEST = "/mnt/data/asr-finetuning/data/training/v3/en/nptel/train_manifest_fixed.json"
OUTPUT_MANIFEST = "/mnt/data/asr-finetuning/data/training/v3/mixed_kn_en_50_50.json"

# --- RATIO CONFIGURATION ---
# Target: English duration = 50% of Kannada duration
EN_RATIO = 1

def create_mixed_manifest():
    kn_entries = []
    en_entries = []
    
    # 1. Load all Kannada entries (100%)
    with open(KANNADA_MANIFEST, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                kn_entries.append(json.loads(line))
    
    total_kn_seconds = sum(entry['duration'] for entry in kn_entries)
    total_kn_hours = total_kn_seconds / 3600
    
    # 2. Calculate target English duration
    target_en_seconds = total_kn_seconds * EN_RATIO
    
    # 3. Load English entries
    with open(ENGLISH_MANIFEST, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                en_entries.append(json.loads(line))
    
    # Shuffle English to ensure unbiased duration-based sampling
    random.shuffle(en_entries)
    
    selected_en = []
    current_en_seconds = 0
    
    for entry in en_entries:
        if current_en_seconds >= target_en_seconds:
            break
        selected_en.append(entry)
        current_en_seconds += entry['duration']
    
    # 4. Combine and Final Shuffle
    final_manifest = kn_entries + selected_en
    random.shuffle(final_manifest)
    
    # 5. Write Output
    with open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f:
        for entry in final_manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Status Report
    print(f"--- Mix Report ---")
    print(f"KN Duration: {total_kn_hours:.2f} hrs ({len(kn_entries)} lines)")
    print(f"EN Duration: {(current_en_seconds / 3600):.2f} hrs ({len(selected_en)} lines)")
    print(f"Total Combined: {((total_kn_seconds + current_en_seconds) / 3600):.2f} hrs")
    print(f"Output saved to: {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    create_mixed_manifest()