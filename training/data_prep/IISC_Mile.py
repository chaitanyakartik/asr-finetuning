import os
import json
import tarfile
import soundfile as sf
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_NAME = "IISc_MILE"
TAR_FILENAME = "mile_kannada_train.tar.gz"
DOWNLOAD_URL = f"https://www.openslr.org/resources/126/{TAR_FILENAME}"

# Paths
OUTPUT_DIR = "/Users/chaitanyakartik/Projects/asr-finetuning/data/training/v2/iisc_mile"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw_extract")
TAR_PATH = os.path.join(OUTPUT_DIR, TAR_FILENAME)

# Ensure directories exist
os.makedirs(RAW_DIR, exist_ok=True)

def run_iisc_mile_pipeline():
    print(f"--- 🚀 Starting Pipeline: {DATASET_NAME} ---")
    
    # 1. CHECK / DOWNLOAD
    if not os.path.exists(TAR_PATH):
        print(f"   📂 File not found locally. Attempting download...")
        print(f"   ⬇️  URL: {DOWNLOAD_URL}")
        exit_code = os.system(f"wget -c {DOWNLOAD_URL} -O {TAR_PATH}")
        if exit_code != 0:
            print("   ❌ Download failed. Please download manually.")
            return
    else:
        print(f"   ✅ Found local archive: {TAR_PATH}")

    # 2. EXTRACT
    flag_file = os.path.join(RAW_DIR, "_extraction_complete")
    if not os.path.exists(flag_file):
        print(f"   📦 Extracting archive (This takes a while)...")
        try:
            with tarfile.open(TAR_PATH, "r:gz") as tar:
                tar.extractall(path=RAW_DIR)
            with open(flag_file, "w") as f: f.write("done")
            print("   ✅ Extraction Complete")
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
            return
    else:
        print("   ⏩ Archive already extracted.")

    # 3. LOCATE FOLDERS (Using your X-Ray intel)
    # Root seems to be 'train'
    train_root = os.path.join(RAW_DIR, "train")
    
    if not os.path.exists(train_root):
        # Fallback: maybe it extracted directly to RAW_DIR?
        train_root = RAW_DIR
    
    print(f"   🔍 Scanning {train_root} for data...")
    
    trans_root = os.path.join(train_root, "trans_files")
    
    # Dynamic Audio Search: Find folder containing .wavs
    audio_root = None
    for root, dirs, files in os.walk(train_root):
        # Check if this folder has wav files
        if any(f.endswith('.wav') for f in files):
            audio_root = root
            break
            
    if not os.path.exists(trans_root):
        print(f"   ❌ Error: Transcript folder not found at {trans_root}")
        return
        
    if not audio_root:
        print(f"   ❌ Error: No folder with .wav files found inside {train_root}")
        return

    print(f"   📍 Audio Source: {audio_root}")
    print(f"   📍 Text Source:  {trans_root}")

    # 4. GENERATE MANIFEST
    manifest_entries = []
    
    # Get list of text files (since text is the ground truth label)
    txt_files = [f for f in os.listdir(trans_root) if f.endswith('.txt')]
    txt_files.sort()
    
    print(f"   Processing {len(txt_files)} pairs...")
    
    for txt_file in tqdm(txt_files):
        try:
            basename = os.path.splitext(txt_file)[0]
            
            # Construct expected wav path
            wav_filename = basename + ".wav"
            wav_path = os.path.join(audio_root, wav_filename)
            
            if not os.path.exists(wav_path):
                # Sometimes mismatch exists, skip
                continue

            # Read Text
            txt_path = os.path.join(trans_root, txt_file)
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                continue

            # Verify Audio Metadata
            # Note: IISc MILE is clean, but verify valid WAV
            info = sf.info(wav_path)
            
            entry = {
                "audio_filepath": os.path.abspath(wav_path),
                "text": text,
                "duration": info.duration,
                "lang": "kn",
                "source": "iisc_mile"
            }
            manifest_entries.append(entry)
            
        except Exception:
            continue

    # 5. SAVE
    manifest_path = os.path.join(OUTPUT_DIR, "train_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"\n✅ Pipeline Complete!")
    print(f"   📄 Manifest: {manifest_path}")
    print(f"   📊 Total Samples: {len(manifest_entries)}")

if __name__ == "__main__":
    run_iisc_mile_pipeline()