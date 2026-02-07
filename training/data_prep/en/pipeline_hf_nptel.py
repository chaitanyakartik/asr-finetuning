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
DATASET_NAME = "NPTEL"
HF_DATASET_ID = "ai4bharat/NPTEL"
HF_CONFIG = "en2indic"
HF_SPLIT="en2indic"
TARGET_HOURS = 700  # Target approximately 700 hours of audio
TARGET_SECONDS = TARGET_HOURS * 3600  # Convert to seconds
OUTPUT_DIR = "data/training/v3/en/nptel"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")

# from datasets import load_dataset
# bhasaanuvaad = load_dataset("ai4bharat/NPTEL", "indic2en", split="hindi")


# Ensure directories exist
os.makedirs(WAV_DIR, exist_ok=True)

def run_nptel_pipeline():
    print(f"--- 🚀 Starting Pipeline: {DATASET_NAME} ---")

    # 1. SETUP & AUTH
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in .env file.")
        return

    try:
        # 2. STREAMING DOWNLOAD - Target ~700 hours
        print(f"   📡 Connecting to Hugging Face ({HF_DATASET_ID})...")
        print(f"   🎯 Target: ~{TARGET_HOURS} hours of audio data")
        print(f"   💾 Cache directory: {CACHE_DIR}")
        print(f"   🔄 Using streaming mode to save disk space")
        
        ds = load_dataset(
            HF_DATASET_ID, 
            HF_CONFIG, 
            split=HF_SPLIT, 
            token=hf_token,
            streaming=True  # Enable streaming - NO cache_dir to prevent downloads!
            # Note: Removed 'trust_remote_code' based on your previous logs
        )
        
        # --- CRITICAL STEP ---
        # NPTEL has chunked audio data in 'chunked_audio_filepath'
        # Cast it to ensure proper decoding at 16kHz
        ds = ds.cast_column("chunked_audio_filepath", Audio(sampling_rate=16000))
        
        manifest_entries = []
        raw_metadata = []
        
        processed_count = 0
        skipped_count = 0
        total_duration = 0.0  # Track total audio duration in seconds

        print(f"   ⬇️  Streaming samples until reaching ~{TARGET_HOURS} hours...")

        for i, item in enumerate(ds):
            # Stop condition - reached target duration
            if total_duration >= TARGET_SECONDS:
                print(f"\n   🎯 Target duration reached: {total_duration/3600:.2f} hours")
                break

            # --- A. INSPECTION (First Item Only) ---
            if i == 0:
                print(f"\n   👀 [INSPECTION] Keys: {list(item.keys())}")
                print(f"   👀 [INSPECTION] Audio Data: {type(item['chunked_audio_filepath'])}")
                print(f"   👀 [INSPECTION] Text: {item.get('text', 'No Text')[:50]}...\n")
                print(f"   👀 [INSPECTION] Pred Text: {item.get('pred_text', 'No Text')[:50]}...\n")

            # --- B. EXTRACT DATA ---
            # Skip if audio is None
            if item['chunked_audio_filepath'] is None:
                skipped_count += 1
                print(f"   ⚠️  Skipping item {i}: No audio data")
                continue
            
            # Access 'chunked_audio_filepath' column (converted to dict by cast_column)
            audio_data_dict = item['chunked_audio_filepath']
            
            audio_array = audio_data_dict['array']
            sr = audio_data_dict['sampling_rate']
            
            # NPTEL uses 'text' for the English transcription (not 'english_sentence')
            text = item.get('text', "")
            
            # --- C. SAVE AUDIO ---
            filename = f"{DATASET_NAME}_{processed_count}.wav"
            file_path = os.path.join(WAV_DIR, filename)
            abs_path = os.path.abspath(file_path)
            
            sf.write(file_path, audio_array, sr)

            # --- D. METADATA COLLECTION ---
            # NPTEL metadata
            raw_metadata.append({
                "original_index": i,
                "filename": filename,
                "sr": sr,
                "text": text,
                "pred_text": item.get('pred_text', ''),
                "course_id": item.get('course_id', ''),
                "video_id": item.get('video_id', ''),
                "start_time": item.get('start_time', 0),
                "alignment_score": item.get('alignment_score', 0)
            })

            # NeMo Manifest Entry
            duration = len(audio_array) / sr
            
            manifest_entry = {
                "audio_filepath": abs_path,
                "text": text,
                "duration": duration,
                "lang": "en",
                "source": "nptel"
            }
            manifest_entries.append(manifest_entry)
            processed_count += 1
            total_duration += duration
            
            # Progress update and CHECKPOINT SAVE every 1000 samples
            if processed_count % 1000 == 0:
                hours_so_far = total_duration / 3600
                print(f"   📊 Processed: {processed_count} samples | Duration: {hours_so_far:.2f}h / {TARGET_HOURS}h")
                
                # Save checkpoint to prevent data loss
                checkpoint_manifest = os.path.join(OUTPUT_DIR, f"train_manifest_checkpoint_{processed_count}.json")
                with open(checkpoint_manifest, "w", encoding="utf-8") as f:
                    for entry in manifest_entries:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write('\n')
                print(f"   💾 Checkpoint saved: {checkpoint_manifest}")
            elif processed_count % 100 == 0:
                hours_so_far = total_duration / 3600
                print(f"   📊 Processed: {processed_count} samples | Duration: {hours_so_far:.2f}h / {TARGET_HOURS}h")

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
        print(f"   ⏱️  Total Duration: {total_duration/3600:.2f} hours ({total_duration:.1f} seconds)")
        print(f"   ⚠️  Skipped: {skipped_count} items (no audio data)")

    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")

if __name__ == "__main__":
    run_nptel_pipeline()