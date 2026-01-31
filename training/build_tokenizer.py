import os
import json
import glob
import logging
import subprocess

# CONFIG
# This gets the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed_data")
TOKENIZER_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training/tokenizers/kn_master")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts") # <--- Target folder

logging.basicConfig(level=logging.INFO)

def main():
    # --- FIX: Create BOTH directories ---
    os.makedirs(TOKENIZER_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCRIPTS_DIR, exist_ok=True) # <--- Added this line
    
    # 1. Gather ALL Text
    print("🔍 Scanning for Kannada manifests...")
    manifests = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*", "train_manifest.json"))
    
    if not manifests:
        print("❌ No manifests found! check data/processed_data/")
        return

    text_corpus_path = os.path.join(TOKENIZER_OUTPUT_DIR, "all_kannada_text.txt")
    print(f"📖 Aggregating text from {len(manifests)} datasets...")
    
    line_count = 0
    with open(text_corpus_path, 'w', encoding='utf-8') as outfile:
        for m_path in manifests:
            print(f"   - Reading {m_path}")
            with open(m_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    data = json.loads(line)
                    text = data.get('text', '').strip()
                    if text:
                        outfile.write(text + "\n")
                        line_count += 1
    
    print(f"✅ Created Corpus: {line_count} lines of text.")

    # 2. Train Tokenizer
    print("🚀 Training SentencePiece BPE Tokenizer...")
    
    tok_script = os.path.join(SCRIPTS_DIR, "process_asr_text_tokenizer.py")
    
    # Download script if missing
    if not os.path.exists(tok_script):
        print("⬇️  Downloading NeMo tokenizer script...")
        subprocess.check_call([
            "wget", "-O", tok_script, 
            "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tokenizers/process_asr_text_tokenizer.py"
        ])

    cmd = [
        "python", tok_script,
        f"--data_file={text_corpus_path}",
        f"--data_root={TOKENIZER_OUTPUT_DIR}",
        "--vocab_size=1024",
        "--tokenizer=spe",
        "--spe_type=bpe",
        "--spe_character_coverage=1.0",
        "--log"
    ]
    
    try:
        subprocess.check_call(cmd)
        print(f"✅ Master Tokenizer saved to: {TOKENIZER_OUTPUT_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Tokenizer training failed: {e}")

if __name__ == "__main__":
    main()
