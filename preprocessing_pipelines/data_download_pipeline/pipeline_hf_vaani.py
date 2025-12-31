import os
import json
import soundfile as sf
from datasets import load_dataset, Audio
from dotenv import load_dotenv

# --- CONFIGURATION ---
DATASET_NAME = "Vaani"
HF_DATASET_ID = "ARTPARK-IISc/Vaani-transcription-part"
HF_CONFIG = "Kannada"  # Verified in your simulation
N_SAMPLES = 10         # Set to -1 for full download
OUTPUT_DIR = "processed_data/Vaani"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")

# Ensure directories exist
os.makedirs(WAV_DIR, exist_ok=True)

def run_vaani_pipeline():
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
        ds = load_dataset(
            HF_DATASET_ID, 
            HF_CONFIG, 
            split="train", 
            streaming=True, 
            token=hf_token
        )
        
        # --- CRITICAL STEP ---
        # Vaani uses 'audio' as the key. We cast it to ensure 16kHz.
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        
        manifest_entries = []
        raw_metadata = []

        print(f"   ⬇️  Streaming & Processing first {N_SAMPLES} samples...")

        for i, item in enumerate(ds):
            # Stop condition
            if N_SAMPLES != -1 and i >= N_SAMPLES:
                break

            # --- A. INSPECTION (First Item Only) ---
            if i == 0:
                print(f"\n   👀 [INSPECTION] Keys: {list(item.keys())}")
                print(f"   👀 [INSPECTION] Audio Data: {type(item['audio'])}")
                print(f"   👀 [INSPECTION] Text: {item.get('transcript', 'No Text')[:50]}...\n")

            # --- B. EXTRACT DATA ---
            # Access 'audio' dictionary
            audio_data_dict = item['audio']
            
            audio_array = audio_data_dict['array']
            sr = audio_data_dict['sampling_rate']
            
            # Vaani uses 'transcript' for the text
            text = item.get('transcript', "")
            
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
                "district": item.get('district', 'unknown'),
                "gender": item.get('gender', 'unknown')
            })

            # NeMo Manifest Entry
            duration = len(audio_array) / sr
            
            manifest_entry = {
                "audio_filepath": abs_path,
                "text": text,
                "duration": duration,
                "lang": "kn",
                "source": "vaani_dialect"
            }
            manifest_entries.append(manifest_entry)

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
    run_vaani_pipeline()