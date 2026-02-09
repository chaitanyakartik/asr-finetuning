import os

CACHE_DIR = "/mnt/data/hf_cache"
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')

import json
import soundfile as sf
from datasets import load_dataset
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

DATASET_NAME = "NPTEL"
HF_DATASET_ID = "ai4bharat/NPTEL"
HF_CONFIG = "en2indic"
HF_SPLIT = "en2indic"
TARGET_HOURS = 700
TARGET_SECONDS = TARGET_HOURS * 3600
OUTPUT_DIR = "data/training/v3/en/nptel"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")

# BLACKLIST THE PROBLEM INDICES
SKIP_INDICES = set(range(16700, 16900))  # Skip the problem area entirely

os.makedirs(WAV_DIR, exist_ok=True)

def run_nptel_pipeline():
    print(f"--- 🚀 Starting Pipeline: {DATASET_NAME} ---")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    ds = load_dataset(HF_DATASET_ID, HF_CONFIG, split=HF_SPLIT, token=hf_token, cache_dir=CACHE_DIR)
    
    manifest_entries = []
    raw_metadata = []
    processed_count = 0
    skipped_count = 0
    total_duration = 0.0
    
    print(f"   📊 Total samples: {len(ds)}")
    print(f"   🚫 Blacklisting indices: {min(SKIP_INDICES)}-{max(SKIP_INDICES)}")
    
    for i in range(len(ds)):
        if total_duration >= TARGET_SECONDS:
            break
        
        # HARD SKIP BLACKLISTED INDICES
        if i in SKIP_INDICES:
            skipped_count += 1
            continue
        
        try:
            # Wrap EVERYTHING
            item = ds[i]
        except:
            skipped_count += 1
            if skipped_count % 100 == 0:
                print(f"   ⚠️  Can't load index {i}, skipped {skipped_count} total")
            continue
        
        try:
            audio_input = item.get('chunked_audio_filepath')
            if not audio_input or hasattr(audio_input, 'read'):
                raise ValueError("Bad audio")
            
            if isinstance(audio_input, str):
                audio_array, sr = sf.read(audio_input)
            elif isinstance(audio_input, dict):
                audio_array = audio_input.get('array')
                sr = audio_input.get('sampling_rate', 16000)
            else:
                raise ValueError("Unknown format")
            
            if audio_array is None or len(audio_array) == 0:
                raise ValueError("Empty")
            
            text = item.get('text', "").strip()
            if not text:
                raise ValueError("No text")
            
            filename = f"{DATASET_NAME}_{processed_count}.wav"
            file_path = os.path.join(WAV_DIR, filename)
            abs_path = os.path.abspath(file_path)
            
            sf.write(file_path, audio_array, sr)
            duration = len(audio_array) / sr
            
            raw_metadata.append({
                "original_index": i,
                "filename": filename,
                "sr": sr,
                "text": text,
            })
            
            manifest_entries.append({
                "audio_filepath": abs_path,
                "text": text,
                "duration": duration,
                "lang": "en",
                "source": "nptel"
            })
            
            processed_count += 1
            total_duration += duration
            
            if processed_count % 100 == 0:
                print(f"   📊 {processed_count} samples | {total_duration/3600:.2f}h / {TARGET_HOURS}h | Skipped: {skipped_count}")
            
            if processed_count % 1000 == 0:
                checkpoint = os.path.join(OUTPUT_DIR, f"checkpoint_{processed_count}.json")
                with open(checkpoint, "w") as f:
                    for entry in manifest_entries:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write('\n')
                print(f"   💾 Checkpoint: {checkpoint}")
                
        except Exception as e:
            skipped_count += 1
            continue
    
    # Save
    manifest_path = os.path.join(OUTPUT_DIR, "train_manifest.json")
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"\n✅ DONE!")
    print(f"   Processed: {processed_count}")
    print(f"   Duration: {total_duration/3600:.2f}h")
    print(f"   Skipped: {skipped_count}")

if __name__ == "__main__":
    run_nptel_pipeline()