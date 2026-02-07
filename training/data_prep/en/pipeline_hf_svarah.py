import os

# CRITICAL: Set cache BEFORE importing datasets to avoid /home disk fill
CACHE_DIR = "/mnt/data/hf_cache"
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')

import json
import soundfile as sf
from datasets import load_dataset, Audio
from dotenv import load_dotenv

# --- CONFIGURATION ---
DATASET_NAME = "Svarah"
HF_DATASET_ID = "ai4bharat/Svarah"
HF_CONFIG = None
HF_SPLIT="train"
N_SAMPLES = -1  # Set to -1 for full download
OUTPUT_DIR = "data/training/v3/en/svarah"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")

# Ensure directories exist
os.makedirs(WAV_DIR, exist_ok=True)

def run_svarah_pipeline():
    print(f"--- 🚀 Starting Pipeline: {DATASET_NAME} ---")

    # 1. SETUP & AUTH
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in .env file.")
        return

    try:
        # 2. FULL DATASET DOWNLOAD
        print(f"   📡 Connecting to Hugging Face ({HF_DATASET_ID})...")
        print(f"   ⚠️  This will download the ENTIRE dataset!")
        print(f"   💾 Cache directory: {CACHE_DIR}")
        
        ds = load_dataset(
            HF_DATASET_ID, 
            HF_CONFIG, 
            split=HF_SPLIT, 
            token=hf_token,
            cache_dir=os.path.join(CACHE_DIR, 'datasets')  # Force cache to /mnt/data
            # Note: Removed 'trust_remote_code' based on your previous logs
        )
        
        # First, let's inspect the dataset structure
        print(f"   🔍 Dataset columns: {ds.column_names}")
        print(f"   🔍 Dataset features: {ds.features}")
        
        # Don't cast - Svarah likely already has audio properly configured
        # If audio_filepath is a string path, we need different handling
        
        manifest_entries = []
        raw_metadata = []
        
        processed_count = 0
        skipped_count = 0

        total_items = len(ds) if N_SAMPLES == -1 else N_SAMPLES
        print(f"   ⬇️  Processing {total_items} samples...")

        for i, item in enumerate(ds):
            # Stop condition
            if N_SAMPLES != -1 and processed_count >= N_SAMPLES:
                break

            # --- A. INSPECTION (First Item Only) ---
            if i == 0:
                print(f"\n   👀 [INSPECTION] Keys: {list(item.keys())}")
                print(f"   👀 [INSPECTION] Audio Data: {type(item['audio_filepath'])}")
                print(f"   👀 [INSPECTION] Text: {item.get('text', 'No Text')[:50]}...\n")
                print(f"   👀 [INSPECTION] Duration from metadata: {item.get('duration', 'N/A')}\n")

            # --- B. EXTRACT DATA ---
            # Skip if audio is None
            if item['audio_filepath'] is None:
                skipped_count += 1
                print(f"   ⚠️  Skipping item {i}: No audio data")
                continue
            
            # Access 'audio_filepath' column (converted to dict by cast_column)
            audio_data_dict = item['audio_filepath']
            
            audio_array = audio_data_dict['array']
            sr = audio_data_dict['sampling_rate']
            
            # Svarah uses 'text' for the English transcription (not 'english_sentence')
            text = item.get('text', "")
            
            # --- C. SAVE AUDIO ---
            filename = f"{DATASET_NAME}_{processed_count}.wav"
            file_path = os.path.join(WAV_DIR, filename)
            abs_path = os.path.abspath(file_path)
            
            sf.write(file_path, audio_array, sr)

            # --- D. METADATA COLLECTION ---
            # Svarah metadata
            raw_metadata.append({
                "original_index": i,
                "filename": filename,
                "sr": sr,
                "text": text,
                "gender": item.get('gender', ''),
                "age_group": item.get('age-group', ''),
                "primary_language": item.get('primary_language', ''),
                "native_place_state": item.get('native_place_state', ''),
                "native_place_district": item.get('native_place_district', ''),
                "highest_qualification": item.get('highest_qualification', ''),
                "job_category": item.get('job_category', ''),
                "occupation_domain": item.get('occupation_domain', '')
            })

            # NeMo Manifest Entry
            duration = len(audio_array) / sr
            
            manifest_entry = {
                "audio_filepath": abs_path,
                "text": text,
                "duration": duration,
                "lang": "en",
                "source": "svarah"
            }
            manifest_entries.append(manifest_entry)
            processed_count += 1

        # --- E. SAVE FILES ---
        meta_path = os.path.join(OUTPUT_DIR, "raw_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(raw_metadata, f, indent=4, ensure_ascii=False)

        manifest_path = os.path.join(OUTPUT_DIR, "train_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        print(f"\n✅ Pipeline Complete!")
        print(f"   📂 Audio: {WAV_DIR}")
        print(f"   📄 Manifest: {manifest_path}")
        print(f"   📊 Processed: {processed_count} items")
        print(f"   ⚠️  Skipped: {skipped_count} items (no audio data)")

    except Exception as e:
        import traceback
        print(f"\n❌ Pipeline Failed: {e}")
        print(f"\n📋 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    run_svarah_pipeline()