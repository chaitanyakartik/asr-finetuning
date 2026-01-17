import os
import zipfile

# --- CONFIGURATION ---
DATASET_NAME = "OpenSLR79_Male"
FILENAME = "kn_in_male.zip"
URL = f"https://www.openslr.org/resources/79/{FILENAME}"

OUTPUT_DIR = "processed_data/OpenSLR79"
FILE_PATH = os.path.join(OUTPUT_DIR, FILENAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def inspect_openslr79():
    print(f"--- 🚀 Starting Inspection: {DATASET_NAME} ---")
    
    # 1. DOWNLOAD
    if not os.path.exists(FILE_PATH):
        print(f"   ⬇️  Downloading {FILENAME}...")
        # Using wget for stability
        exit_code = os.system(f'wget -c "{URL}" -O {FILE_PATH}')
        if exit_code != 0:
            print("   ❌ Download failed.")
            return
    else:
        print(f"   ✅ Found local file: {FILE_PATH}")

    # 2. X-RAY (PEEK INSIDE ZIP)
    print(f"\n--- 🔍 X-Raying Structure ---")
    try:
        with zipfile.ZipFile(FILE_PATH, 'r') as zf:
            print("   (Reading file list...)")
            
            # Print first 25 items to capture folder structure + some files
            for i, name in enumerate(zf.namelist()):
                print(f"   📄 {name}")
                if i >= 25:
                    break
                    
        print("\n--- ✅ Inspection Done ---")
        print("   Please paste the output above so I can write the pipeline.")

    except Exception as e:
        print(f"   ❌ Error reading zip: {e}")

if __name__ == "__main__":
    inspect_openslr79()