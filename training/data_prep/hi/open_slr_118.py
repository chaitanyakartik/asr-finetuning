import os
import zipfile
import tarfile
import json
import soundfile as sf
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_NAME = "Gramvaani"
DATASET_URL = "https://asr.iitm.ac.in/Gramvaani/NEW/GV_Train_100h.tar.gz"
FILENAME = "GV_Train_100h.tar.gz"
OUTPUT_DIR = "/Users/chaitanyakartik/Projects/asr-finetuning/evaluation/benchmarking/data/v2/Gramvaani"

def run_gramvaani_pipeline():
    print(f"--- 🚀 Starting Pipeline: {DATASET_NAME} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    manifest_entries = []
    
    # 1. DOWNLOAD
    tar_path = os.path.join(OUTPUT_DIR, FILENAME)
    if not os.path.exists(tar_path):
        print(f"   ⬇️  Downloading {FILENAME}...")
        print(f"   📍 From: {DATASET_URL}")
        # Use -c to resume if connection breaks
        exit_code = os.system(f'wget -c "{DATASET_URL}" -O {tar_path}')
        if exit_code != 0:
            print("   ❌ Download failed.")
            return
    else:
        print(f"   ✅ Found local archive: {FILENAME}")

    # 2. EXTRACT
    extract_path = os.path.join(OUTPUT_DIR, "raw_extract")
    flag_file = os.path.join(extract_path, "_extracted.done")
    
    if not os.path.exists(flag_file):
        print(f"   📦 Extracting to {extract_path}...")
        try:
            os.makedirs(extract_path, exist_ok=True)
            with tarfile.open(tar_path, 'r:gz') as tf:
                tf.extractall(extract_path)
            with open(flag_file, "w") as f: f.write("done")
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
            return
    else:
        print(f"   ⏩ Already extracted.")
    
    # 3. LOCATE METADATA
    # Look for transcript file (could be .tsv, .txt, or other formats)
    print(f"   🔍 Scanning for transcript files...")
    tsv_path = None
    
    # Common transcript file names
    possible_names = ["line_index.tsv", "transcripts.tsv", "transcript.txt", "text", "train.tsv"]
    
    for root, dirs, files in os.walk(extract_path):
        for possible_name in possible_names:
            if possible_name in files:
                tsv_path = os.path.join(root, possible_name)
                print(f"   ✅ Found transcript file: {os.path.relpath(tsv_path, extract_path)}")
                break
        if tsv_path:
            break
    
    if not tsv_path:
        print(f"   ⚠️ No standard transcript file found. Listing directory structure...")
        for root, dirs, files in os.walk(extract_path):
            level = root.replace(extract_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        print(f"\n   ❌ Error: Could not find transcript file. Please check the structure above.")
        return

    # 4. PARSE & VALIDATE
    print(f"   📖 Parsing {os.path.basename(tsv_path)}...")
    
    processed_count = 0
    wav_root = os.path.dirname(tsv_path) # Wavs are usually with the TSV
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
                
            filename = parts[0]
            text = parts[1]
            
            # Construct absolute path (try with and without .wav extension)
            if not filename.endswith('.wav'):
                wav_path = os.path.join(wav_root, filename + ".wav")
            else:
                wav_path = os.path.join(wav_root, filename)
            
            # Verify existence (fast check)
            if not os.path.exists(wav_path):
                # Try looking in subdirectories
                for root, dirs, files in os.walk(extract_path):
                    potential_path = os.path.join(root, os.path.basename(filename))
                    if not potential_path.endswith('.wav'):
                        potential_path += '.wav'
                    if os.path.exists(potential_path):
                        wav_path = potential_path
                        break
                
                if not os.path.exists(wav_path):
                    continue
            
            try:
                # Verify Valid Audio & Get Duration
                info = sf.info(wav_path)
                
                entry = {
                    "audio_filepath": os.path.abspath(wav_path),
                    "text": text,
                    "duration": info.duration,
                    "lang": "hi",
                    "source": "gramvaani_100h"
                }
                manifest_entries.append(entry)
                processed_count += 1
            except Exception:
                # Skip corrupt files
                pass
    
    print(f"      + Added {processed_count} samples")

    # 5. SAVE MANIFEST
    if manifest_entries:
        manifest_path = os.path.join(OUTPUT_DIR, "train_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        print(f"\n✅ Pipeline Complete!")
        print(f"   📄 Manifest: {manifest_path}")
        print(f"   📊 Total Samples: {len(manifest_entries)}")
    else:
        print("\n❌ No samples processed.")

if __name__ == "__main__":
    run_gramvaani_pipeline()