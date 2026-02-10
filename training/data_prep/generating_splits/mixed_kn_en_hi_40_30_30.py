import json
import random
import os

# --- PATH CONFIGURATION ---
KANNADA_MANIFEST = "/mnt/data/asr-finetuning/data/training/v2.1/master_manifest.json"
ENGLISH_MANIFEST = "/mnt/data/asr-finetuning/data/training/v3/en/nptel/train_manifest_fixed.json"
HINDI_MANIFEST = "/mnt/data/asr-finetuning/data/training/v3/hi/train_manifest_combined.json"
OUTPUT_MANIFEST = "/mnt/data/asr-finetuning/data/training/v3/mixed_kn_en_hi_30_30_40.json"

# --- RATIO CONFIGURATION ---
# Based on Kannada (40%): English (30%) and Hindi (30%)
# Calculation: Target = (Kannada_Duration / 0.40) * 0.30
EN_RATIO = 0.75  # 0.30 / 0.40
HI_RATIO = 0.75  # 0.30 / 0.40

def create_triple_mix_manifest():
    kn_entries = []
    en_entries = []
    hi_entries = []
    
    # 1. Load all Kannada entries (Base: 40% of the total mix)
    with open(KANNADA_MANIFEST, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                kn_entries.append(json.loads(line))
    
    total_kn_seconds = sum(entry['duration'] for entry in kn_entries)
    
    # 2. Calculate target durations for EN and HI
    target_en_seconds = total_kn_seconds * EN_RATIO
    target_hi_seconds = total_kn_seconds * HI_RATIO
    
    # 3. Process Secondary Datasets
    def sample_dataset(path, target_seconds):
        entries = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        random.shuffle(entries)
        selected = []
        current_seconds = 0
        for entry in entries:
            if current_seconds >= target_seconds:
                break
            selected.append(entry)
            current_seconds += entry['duration']
        return selected, current_seconds

    selected_en, actual_en_seconds = sample_dataset(ENGLISH_MANIFEST, target_en_seconds)
    selected_hi, actual_hi_seconds = sample_dataset(HINDI_MANIFEST, target_hi_seconds)
    
    # 4. Combine and Final Shuffle
    final_manifest = kn_entries + selected_en + selected_hi
    random.shuffle(final_manifest)
    
    # 5. Write Output
    with open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f:
        for entry in final_manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Status Report
    def to_hrs(sec): return sec / 3600

    total_sec = total_kn_seconds + actual_en_seconds + actual_hi_seconds
    print(f"--- Triple Mix Report (Target 40/30/30) ---")
    print(f"KN (40%): {to_hrs(total_kn_seconds):.2f} hrs | Actual: {(total_kn_seconds/total_sec)*100:.1f}%")
    print(f"EN (30%): {to_hrs(actual_en_seconds):.2f} hrs | Actual: {(actual_en_seconds/total_sec)*100:.1f}%")
    print(f"HI (30%): {to_hrs(actual_hi_seconds):.2f} hrs | Actual: {(actual_hi_seconds/total_sec)*100:.1f}%")
    print(f"Total Combined: {to_hrs(total_sec):.2f} hrs")
    print(f"Output saved to: {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    create_triple_mix_manifest()