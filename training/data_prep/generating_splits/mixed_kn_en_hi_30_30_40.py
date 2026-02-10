import json
import random
import os

# --- PATH CONFIGURATION ---
KANNADA_MANIFEST = "/mnt/data/asr-finetuning/data/training/v2.1/master_manifest.json"
ENGLISH_MANIFEST = "/mnt/data/asr-finetuning/data/training/v3/en/nptel/train_manifest_fixed.json"
HINDI_MANIFEST = "/mnt/data/asr-finetuning/data/training/v3/hi/train_manifest_combined.json"
OUTPUT_MANIFEST = "/mnt/data/asr-finetuning/data/training/v3/mixed_kn_en_hi_40_30_30.json"

# --- RATIO CONFIGURATION ---
# Base: Kannada (30% of total)
# Target: English = 30%, Hindi = 40%
# Calculations relative to Kannada:
# EN = (30% / 30%) * KN_Duration = 1.0 * KN_Duration
# HI = (40% / 30%) * KN_Duration = 1.333 * KN_Duration
EN_RATIO = 1.0
HI_RATIO = 1.333333 

def create_specific_mix():
    kn_entries = []
    
    # 1. Load all Kannada entries (The 30% anchor)
    if not os.path.exists(KANNADA_MANIFEST):
        print(f"❌ Error: Kannada manifest not found at {KANNADA_MANIFEST}")
        return

    with open(KANNADA_MANIFEST, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                kn_entries.append(json.loads(line))
    
    total_kn_seconds = sum(entry['duration'] for entry in kn_entries)
    
    # 2. Set targets
    target_en_seconds = total_kn_seconds * EN_RATIO
    target_hi_seconds = total_kn_seconds * HI_RATIO
    
    def sample_dataset(path, target_seconds, lang_label):
        if not os.path.exists(path):
            print(f"⚠️ Warning: {lang_label} manifest not found. Skipping.")
            return [], 0
            
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
        
        if current_seconds < target_seconds:
            print(f"⚠️ Warning: Not enough {lang_label} data to hit target. "
                  f"Got {current_seconds/3600:.2f}/{target_seconds/3600:.2f} hrs")
                  
        return selected, current_seconds

    # 3. Sample English and Hindi
    selected_en, actual_en_seconds = sample_dataset(ENGLISH_MANIFEST, target_en_seconds, "English")
    selected_hi, actual_hi_seconds = sample_dataset(HINDI_MANIFEST, target_hi_seconds, "Hindi")
    
    # 4. Combine and Shuffle
    final_manifest = kn_entries + selected_en + selected_hi
    random.shuffle(final_manifest)
    
    # 5. Write Output
    with open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f:
        for entry in final_manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Status Report
    def to_hrs(sec): return sec / 3600
    total_sec = total_kn_seconds + actual_en_seconds + actual_hi_seconds
    
    print(f"--- 📊 Final Mix Report (30/30/40) ---")
    print(f"KN (30% Target): {to_hrs(total_kn_seconds):.2f} hrs | Actual: {(total_kn_seconds/total_sec)*100:.1f}%")
    print(f"EN (30% Target): {to_hrs(actual_en_seconds):.2f} hrs | Actual: {(actual_en_seconds/total_sec)*100:.1f}%")
    print(f"HI (40% Target): {to_hrs(actual_hi_seconds):.2f} hrs | Actual: {(actual_hi_seconds/total_sec)*100:.1f}%")
    print(f"Total Duration: {to_hrs(total_sec):.2f} hrs")
    print(f"Saved to: {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    create_specific_mix()