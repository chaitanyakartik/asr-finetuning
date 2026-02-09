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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# --- CONFIGURATION ---
DATASET_NAME = "Kathbath"
HF_DATASET_ID = "ai4bharat/kathbath"
HF_CONFIG = "hindi"  # Verified in your previous inspection
N_SAMPLES = -1        # Set to -1 for full download
OUTPUT_DIR = "data/training/v3/hi/kathbath"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")
NUM_WORKERS = 16      # Parallel workers for WAV file writing

# Ensure directories exist
os.makedirs(WAV_DIR, exist_ok=True)

def run_kathbath_pipeline():
    print(f"--- 🚀 Starting Pipeline: {DATASET_NAME} ---")

    # 1. SETUP & AUTH
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in .env file.")
        return

    try:
        # 2. STREAMING INGESTION
        print(f"   📡 Connecting to Hugging Face ({HF_DATASET_ID})...")
        print(f"   💾 Cache directory: {CACHE_DIR}")
        
        ds = load_dataset(
            HF_DATASET_ID, 
            HF_CONFIG, 
            split="train", 
            streaming=False, 
            token=hf_token,
            cache_dir=os.path.join(CACHE_DIR, 'datasets'),
            trust_remote_code=True # Kathbath sometimes needs this for the builder script
        )
        
        # --- CRITICAL STEP ---
        # Kathbath also uses 'audio_filepath'. We must cast it to Audio() 
        # to trigger the download and decoding.
        ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
        
        manifest_entries = []
        raw_metadata = []
        lock = threading.Lock()
        
        # Progress tracking
        PROGRESS_INTERVAL = 100
        SAVE_INTERVAL = 1000  # Save every 1000 samples
        BATCH_SIZE = 50       # Process in batches for better throughput

        print(f"   ⬇️  Processing samples with {NUM_WORKERS} workers...")
        print(f"   📊 Progress will be logged every {PROGRESS_INTERVAL} samples\n")
        
        def process_sample(i, item):
            """Process a single sample - extract, save audio, create manifest entry"""
            try:
                audio_data_dict = item['audio_filepath']
                audio_array = audio_data_dict['array']
                sr = audio_data_dict['sampling_rate']
                text = item.get('text', "")
                
                # Save audio
                filename = f"{DATASET_NAME}_{i}.wav"
                file_path = os.path.join(WAV_DIR, filename)
                abs_path = os.path.abspath(file_path)
                sf.write(file_path, audio_array, sr)
                
                # Create metadata
                metadata = {
                    "original_index": i,
                    "filename": filename,
                    "sr": sr,
                    "speaker_id": item.get('speaker_id', 'unknown'),
                    "gender": item.get('gender', 'unknown')
                }
                
                # Create manifest entry
                duration = len(audio_array) / sr
                manifest_entry = {
                    "audio_filepath": abs_path,
                    "text": text,
                    "duration": duration,
                    "lang": "kn",
                    "source": "kathbath_clean"
                }
                
                return (i, metadata, manifest_entry, None)
            except Exception as e:
                return (i, None, None, str(e))
        
        # Collect items for batch processing
        batch = []
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for i, item in enumerate(ds):
                # Stop condition
                if N_SAMPLES != -1 and i >= N_SAMPLES:
                    break
                
                # Inspection on first item
                if i == 0:
                    print(f"   👀 [INSPECTION] Keys: {list(item.keys())}")
                    print(f"   👀 [INSPECTION] Audio Data: {type(item['audio_filepath'])}")
                    print(f"   👀 [INSPECTION] Text: {item.get('text', 'No Text')[:50]}...\n")
                
                batch.append((i, item))
                
                # Process batch when full
                if len(batch) >= BATCH_SIZE:
                    futures = [executor.submit(process_sample, idx, itm) for idx, itm in batch]
                    
                    for future in as_completed(futures):
                        idx, metadata, manifest_entry, error = future.result()
                        
                        if error:
                            print(f"   ⚠️  Error processing sample {idx}: {error}")
                        else:
                            raw_metadata.append(metadata)
                            manifest_entries.append(manifest_entry)
                        
                        processed_count += 1
                        
                        # Progress logging
                        if processed_count % PROGRESS_INTERVAL == 0:
                            print(f"   ✓ Processed {processed_count} samples...")
                        
                        # Periodic checkpoint save
                        if processed_count % SAVE_INTERVAL == 0:
                            print(f"   💾 Checkpoint save at {processed_count} samples...")
                            manifest_path = os.path.join(OUTPUT_DIR, "train_manifest.json")
                            with lock:
                                with open(manifest_path, "w", encoding="utf-8") as f:
                                    for entry in sorted(manifest_entries, key=lambda x: x['audio_filepath']):
                                        json.dump(entry, f, ensure_ascii=False)
                                        f.write('\n')
                    
                    batch = []
            
            # Process remaining items in final batch
            if batch:
                futures = [executor.submit(process_sample, idx, itm) for idx, itm in batch]
                
                for future in as_completed(futures):
                    idx, metadata, manifest_entry, error = future.result()
                    
                    if error:
                        print(f"   ⚠️  Error processing sample {idx}: {error}")
                    else:
                        raw_metadata.append(metadata)
                        manifest_entries.append(manifest_entry)
                    
                    processed_count += 1
                    
                    if processed_count % PROGRESS_INTERVAL == 0:
                        print(f"   ✓ Processed {processed_count} samples...")

        # --- E. FINAL SAVE ---
        print(f"\n   💾 Saving final files...")
        
        # Sort by index before saving
        raw_metadata.sort(key=lambda x: x['original_index'])
        manifest_entries.sort(key=lambda x: x['audio_filepath'])
        
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
        print(f"   📊 Processed: {len(manifest_entries)} items")

    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")

if __name__ == "__main__":
    run_kathbath_pipeline()
