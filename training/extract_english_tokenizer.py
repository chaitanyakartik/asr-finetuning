import os
import tarfile
import logging
from nemo.collections.asr.models import EncDecRNNTBPEModel

# CONFIG
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training/tokenizers/en/tokenizer_spe_bpe_v128")

logging.basicConfig(level=logging.INFO)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("⬇️  Step 1: triggering download (or finding cache)...")
    # This ensures the .nemo file is present in the cache
    model = EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_small")
    
    # NeMo models usually have a hidden attribute pointing to their source file
    # Or we can infer it from the class logic, but let's look at the object
    nemo_file_path = None
    
    # 1. Try to find the path from the object (works in newer NeMo)
    if hasattr(model, 'nemo_file_path'):
        nemo_file_path = model.nemo_file_path
    
    # 2. If that fails, look in the standard cache directory
    if not nemo_file_path:
        print("   Path attribute missing. Searching standard cache...")
        home = os.path.expanduser("~")
        cache_root = os.path.join(home, ".cache/torch/NeMo")
        # Recursively search for the specific file
        for root, dirs, files in os.walk(cache_root):
            for file in files:
                if file == "stt_en_conformer_transducer_small.nemo":
                    nemo_file_path = os.path.join(root, file)
                    break
            if nemo_file_path: break

    if not nemo_file_path or not os.path.exists(nemo_file_path):
        print("❌ CRITICAL: Could not locate the .nemo file on disk.")
        return

    print(f"📦 Found checkpoint at: {nemo_file_path}")

    # Step 3: Unzip it
    print("🔓 Step 2: Extracting tokenizer.model from archive...")
    found_tokenizer = False
    
    try:
        with tarfile.open(nemo_file_path, "r:gz") as tar:
            for member in tar.getmembers():
                # The file is usually named 'tokenizer.model' inside the root of the tar
                if member.name.endswith("tokenizer.model"):
                    print(f"   Found internal file: {member.name}")
                    
                    # Extract to our target directory
                    member.name = "tokenizer.model" # Rename to generic name
                    tar.extract(member, path=OUTPUT_DIR)
                    found_tokenizer = True
                    break
        
        if found_tokenizer:
            print(f"✅ Success! Tokenizer saved to: {os.path.join(OUTPUT_DIR, 'tokenizer.model')}")
        else:
            print("❌ Error: The .nemo archive exists but does not contain 'tokenizer.model'.")
            print("   This is rare. The model might use a different tokenizer type.")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    main()
