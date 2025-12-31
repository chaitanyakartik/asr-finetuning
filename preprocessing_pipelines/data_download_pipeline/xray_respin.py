import os
import tarfile
import requests
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_NAME = "ReSPIN_Test"
FILENAME = "IISc_RESPIN_test_kn.tar.gz"
# The specific signed URL you provided for the TEST set
URL = "https://objectstore.e2enetworks.net/iisc-spire-corpora/respin/kannada/IISc_RESPIN_test_kn.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20251231%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251231T080141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=3efff08fde40f9232a801ac9d587deb24e07003fd6109b880936d36df77de576"

OUTPUT_DIR = "processed_data/ReSPIN"
TAR_PATH = os.path.join(OUTPUT_DIR, FILENAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def inspect_respin():
    print(f"--- 🚀 Starting Inspection: {DATASET_NAME} ---")
    
    # 1. DOWNLOAD
    if not os.path.exists(TAR_PATH):
        print(f"   ⬇️  Downloading {FILENAME}...")
        try:
            # We use system wget because the URL has complex query params (signatures)
            # enclosing URL in quotes is critical here
            exit_code = os.system(f'wget "{URL}" -O {TAR_PATH}')
            if exit_code != 0:
                print("   ❌ Download failed.")
                return
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return
    else:
        print(f"   ✅ Found local file: {TAR_PATH}")

    # 2. X-RAY (PEEK INSIDE)
    print(f"\n--- 🔍 X-Raying Structure ---")
    try:
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            print("   (Reading headers...)")
            for i, member in enumerate(tar):
                print(f"   📄 {member.name}")
                if i >= 20:
                    break
        print("\n--- ✅ Inspection Done ---")
        print("   Please paste the output above so I can write the pipeline.")

    except Exception as e:
        print(f"   ❌ Error reading tar: {e}")

if __name__ == "__main__":
    inspect_respin()