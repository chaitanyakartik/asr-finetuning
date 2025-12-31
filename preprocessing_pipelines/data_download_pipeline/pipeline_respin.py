import os
import json
import tarfile
import soundfile as sf
from tqdm import tqdm

# --- CONFIGURATION ---
# We map the "Split Name" to your specific Signed URL
DATASETS = {
    "test": "https://objectstore.e2enetworks.net/iisc-spire-corpora/respin/kannada/IISc_RESPIN_test_kn.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20251231%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251231T080141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=3efff08fde40f9232a801ac9d587deb24e07003fd6109b880936d36df77de576",
    "dev": "https://objectstore.e2enetworks.net/iisc-spire-corpora/respin/kannada/IISc_RESPIN_dev_kn.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20251231%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251231T080141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=20d4daf507397d9eb88cbdf8aabf2bdd67e0e118e65017840114e0a80348ec8a",
    "train_small": "https://objectstore.e2enetworks.net/iisc-spire-corpora/respin/kannada/IISc_RESPIN_train_kn_small.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20251231%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251231T080141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=1e38e02ef45eca07b33c4e782dba3347c72e0246405b1aec5d21a230c9d6c409",
    "train_clean": "https://objectstore.e2enetworks.net/iisc-spire-corpora/respin/kannada/IISc_RESPIN_train_kn_clean.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20251231%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251231T080141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=52b10d80a65663c929aa3e56b0018362e9a0f59ec563907b9df9d941f1d90680",
    # Uncomment these if you have space and want the noisy/semi-noisy data too
    # "train_seminoisy": "https://objectstore.e2enetworks.net/iisc-spire-corpora/respin/kannada/IISc_RESPIN_train_kn_seminoisy.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20251231%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251231T080141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=76bcf8264354eaa102f79026991951fb7f2992bad49fe9515c87ad4655f404c3",
    # "train_noisy": "https://objectstore.e2enetworks.net/iisc-spire-corpora/respin/kannada/IISc_RESPIN_train_kn_noisy.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20251231%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251231T080141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c74888e15f7d39b3038039aa558bb99c339dbd2b23ab07479cdb66a035066f91"
}

BASE_DIR = "processed_data/ReSPIN"
os.makedirs(BASE_DIR, exist_ok=True)

def process_respin():
    print("--- 🚀 Starting ReSPIN Master Pipeline ---")

    for split_name, url in DATASETS.items():
        print(f"\n📂 Processing Split: {split_name.upper()}")
        
        # 1. SETUP PATHS
        # Filename logic: Extract actual filename from URL (before the ? query params)
        filename = url.split("?")[0].split("/")[-1]
        tar_path = os.path.join(BASE_DIR, filename)
        extract_path = os.path.join(BASE_DIR, "extracted", split_name)
        os.makedirs(extract_path, exist_ok=True)

        # 2. DOWNLOAD
        if not os.path.exists(tar_path):
            print(f"   ⬇️  Downloading {filename}...")
            # Quotes around URL are mandatory due to special chars like & and ?
            exit_code = os.system(f'wget "{url}" -O {tar_path}')
            if exit_code != 0:
                print(f"   ❌ Failed to download {split_name}")
                continue
        else:
            print(f"   ✅ Archive exists: {filename}")

        # 3. EXTRACT
        flag_file = os.path.join(extract_path, "_extracted.done")
        if not os.path.exists(flag_file):
            print(f"   📦 Extracting (this takes time)...")
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=extract_path)
                with open(flag_file, "w") as f: f.write("done")
            except Exception as e:
                print(f"   ❌ Extraction failed: {e}")
                continue
        
        # 4. PARSE & GENERATE MANIFEST
        print(f"   🔄 Parsing Audio-Transcript Mapping...")
        manifest_entries = []
        
        # We walk through the folders to find "Speaker Folders" (folders that contain a .txt file)
        # Structure: root/D4/SpeakerID/
        for root, dirs, files in os.walk(extract_path):
            
            # Find the transcript file in this folder (if any)
            txt_files = [f for f in files if f.endswith(".txt")]
            
            if len(txt_files) == 1:
                # Found a Speaker Folder!
                txt_filename = txt_files[0]
                txt_path = os.path.join(root, txt_filename)
                
                # A. Parse the Transcript File
                # Format Assumption: "FILENAME <tab/space> TRANSCRIPT"
                # We load it into a dict: { "filename": "text" }
                transcript_map = {}
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) == 2:
                            # ReSPIN text files usually have the full audio filename (without extension) as key
                            key = parts[0]
                            content = parts[1]
                            transcript_map[key] = content
                            
                # B. Match with Wav Files
                wav_files = [f for f in files if f.endswith(".wav")]
                for wav in wav_files:
                    wav_id = os.path.splitext(wav)[0]
                    
                    if wav_id in transcript_map:
                        full_wav_path = os.path.join(root, wav)
                        text = transcript_map[wav_id]
                        
                        try:
                            # Get duration
                            info = sf.info(full_wav_path)
                            
                            entry = {
                                "audio_filepath": os.path.abspath(full_wav_path),
                                "text": text,
                                "duration": info.duration,
                                "lang": "kn",
                                "source": f"respin_{split_name}"
                            }
                            manifest_entries.append(entry)
                        except Exception:
                            pass
        
        # 5. SAVE MANIFEST
        # We save separate manifests for each split (train vs test vs dev)
        manifest_path = os.path.join(BASE_DIR, f"{split_name}_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"   ✅ Done! Saved {len(manifest_entries)} items to {manifest_path}")

if __name__ == "__main__":
    process_respin()