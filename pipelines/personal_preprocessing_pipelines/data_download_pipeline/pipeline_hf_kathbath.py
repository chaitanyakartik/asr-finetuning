import os
import json
import soundfile as sf
from datasets import load_dataset, Audio
from dotenv import load_dotenv

# --- CONFIGURATION ---
DATASET_NAME = "Kathbath"
HF_DATASET_ID = "ai4bharat/kathbath"
HF_CONFIG = "kannada"  # Verified in your previous inspection
N_SAMPLES = 50         # Set to -1 for full download
OUTPUT_DIR = "/Users/chaitanyakartik/Projects/asr-finetuning/evaluation/benchmarking/data/v2/Kathbath"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")

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
        ds = load_dataset(
            HF_DATASET_ID, 
            HF_CONFIG, 
            split="valid", 
            streaming=True, 
            token=hf_token,
            trust_remote_code=True # Kathbath sometimes needs this for the builder script
        )
        
        # --- CRITICAL STEP ---
        # Kathbath also uses 'audio_filepath'. We must cast it to Audio() 
        # to trigger the download and decoding.
        ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
        
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
                print(f"   👀 [INSPECTION] Audio Data: {type(item['audio_filepath'])}")
                print(f"   👀 [INSPECTION] Text: {item.get('text', 'No Text')[:50]}...\n")

            # --- B. EXTRACT DATA ---
            # Access 'audio_filepath' (which is now a dict after casting)
            audio_data_dict = item['audio_filepath']
            
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
                "speaker_id": item.get('speaker_id', 'unknown'),
                "gender": item.get('gender', 'unknown')
            })

            # NeMo Manifest Entry
            duration = len(audio_array) / sr
            
            manifest_entry = {
                "audio_filepath": abs_path,
                "text": text,
                "duration": duration,
                "lang": "kn",
                "source": "kathbath_clean"
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
    run_kathbath_pipeline()
