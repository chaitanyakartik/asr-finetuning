import os
import zipfile
import json
import soundfile as sf
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_NAME = "OpenSLR79"
BASE_URL = "https://www.openslr.org/resources/79/"
OUTPUT_DIR = "processed_data/OpenSLR79"

# Sources for both genders
SOURCES = [
    {
        "name": "male",
        "filename": "kn_in_male.zip",
        "url": BASE_URL + "kn_in_male.zip"
    },
    {
        "name": "female",
        "filename": "kn_in_female.zip",
        "url": BASE_URL + "kn_in_female.zip"
    }
]

def run_openslr79_pipeline():
    print(f"--- 🚀 Starting Pipeline: {DATASET_NAME} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    manifest_entries = []

    for source in SOURCES:
        print(f"\n🔹 Processing: Kannada {source['name'].capitalize()}")
        
        # 1. DOWNLOAD
        zip_path = os.path.join(OUTPUT_DIR, source['filename'])
        if not os.path.exists(zip_path):
            print(f"   ⬇️  Downloading {source['filename']}...")
            # Use -c to resume if connection breaks
            exit_code = os.system(f'wget -c "{source["url"]}" -O {zip_path}')
            if exit_code != 0:
                print("   ❌ Download failed.")
                continue
        else:
            print(f"   ✅ Found local archive: {source['filename']}")

        # 2. EXTRACT
        # We extract into specific folders (male/female) to keep them organized
        extract_path = os.path.join(OUTPUT_DIR, "raw_extract", source['name'])
        flag_file = os.path.join(extract_path, "_extracted.done")
        
        if not os.path.exists(flag_file):
            print(f"   📦 Extracting to {extract_path}...")
            try:
                os.makedirs(extract_path, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(extract_path)
                with open(flag_file, "w") as f: f.write("done")
            except Exception as e:
                print(f"   ❌ Extraction failed: {e}")
                continue
        else:
            print(f"   ⏩ Already extracted.")
        
        # 3. LOCATE METADATA
        # Standard OpenSLR file is 'line_index.tsv'
        tsv_path = os.path.join(extract_path, "line_index.tsv")
        
        if not os.path.exists(tsv_path):
            print(f"   ⚠️ 'line_index.tsv' not in root. Scanning subfolders...")
            for root, dirs, files in os.walk(extract_path):
                if "line_index.tsv" in files:
                    tsv_path = os.path.join(root, "line_index.tsv")
                    break
        
        if not os.path.exists(tsv_path):
            print(f"   ❌ Error: Could not find transcripts for {source['name']}")
            continue

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
                
                # Construct absolute path
                wav_path = os.path.join(wav_root, filename + ".wav")
                
                # Verify existence (fast check)
                if not os.path.exists(wav_path):
                    continue
                
                try:
                    # Verify Valid Audio & Get Duration
                    info = sf.info(wav_path)
                    
                    entry = {
                        "audio_filepath": os.path.abspath(wav_path),
                        "text": text,
                        "duration": info.duration,
                        "lang": "kn",
                        "source": f"openslr79_{source['name']}"
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
    run_openslr79_pipeline()