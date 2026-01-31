import os
import json
import soundfile as sf
from datasets import load_dataset, Audio
from dotenv import load_dotenv

# --- CONFIGURATION ---
DATASET_NAME = "Shrutilipi"
HF_DATASET_ID = "ai4bharat/Shrutilipi"
HF_CONFIG = "kannada"

### Set to -1 for full download
N_SAMPLES = -1  
OUTPUT_DIR = "data/training/v2/shrutilipi"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")

# Ensure directories exist
os.makedirs(WAV_DIR, exist_ok=True)

def run_shrutilipi_pipeline():
    print(f"--- 🚀 Starting Pipeline: {DATASET_NAME} ---")

    # 1. SETUP & AUTH
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in .env file.")
        return

    try:
        # 2. LOAD FULL DATASET (NO STREAMING)
        print(f"   📡 Downloading full dataset from Hugging Face ({HF_DATASET_ID})...")
        print(f"   ⏳ This may take a few minutes for the initial download...")
        
        ds = load_dataset(
            HF_DATASET_ID, 
            HF_CONFIG, 
            split="train", 
            token=hf_token
        )
        
        # --- CRITICAL FIX ---
        # The dataset has 'audio_filepath' as the key. 
        # We MUST cast it to Audio() so Hugging Face downloads and decodes it for us.
        # We also force 16kHz here to be safe.
        ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
        
        print(f"   ✅ Dataset loaded: {len(ds)} total samples")
        
        # Determine how many samples to process
        num_to_process = len(ds) if N_SAMPLES == -1 else min(N_SAMPLES, len(ds))
        print(f"   🔄 Processing {num_to_process} samples...")
        
        manifest_entries = []
        raw_metadata = []

        for i in range(num_to_process):
            item = ds[i]

            # --- A. INSPECTION (First Item Only) ---
            if i == 0:
                print(f"\n   👀 [INSPECTION] Keys: {list(item.keys())}")
                # Now that we cast it, 'audio_filepath' should be a dict with 'array'
                print(f"   👀 [INSPECTION] Audio Data: {type(item['audio_filepath'])}")
                print(f"   👀 [INSPECTION] Text: {item.get('text', 'No Text')[:50]}...\n")

            # --- B. EXTRACT DATA ---
            # FIX: Access 'audio_filepath' instead of 'audio'
            audio_data_dict = item['audio_filepath']
            
            # The cast_column ensures this dict has 'array' and 'sampling_rate'
            audio_array = audio_data_dict['array']
            sr = audio_data_dict['sampling_rate']
            
            text = item.get('text', "")
            
            # --- C. SAVE AUDIO ---
            filename = f"{DATASET_NAME}_{i}.wav"
            file_path = os.path.join(WAV_DIR, filename)
            abs_path = os.path.abspath(file_path)
            
            sf.write(file_path, audio_array, sr)

            # --- D. METADATA COLLECTION ---
            raw_metadata.append({
                "original_index": i,
                "filename": filename,
                "sr": sr,
                "text_snippet": text[:30]
            })

            # NeMo Manifest Entry
            duration = len(audio_array) / sr
            
            manifest_entry = {
                "audio_filepath": abs_path,
                "text": text,
                "duration": duration,
                "lang": "kn",
                "source": "shrutilipi_news"
            }
            manifest_entries.append(manifest_entry)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{num_to_process} samples...")

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
        print(f"   📊 Processed: {len(manifest_entries)} items")

    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")

if __name__ == "__main__":
    run_shrutilipi_pipeline()