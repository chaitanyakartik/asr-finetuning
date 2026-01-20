import os
import sentencepiece as spm
import logging

# CONFIG
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
EN_TOK_DIR = os.path.join(PROJECT_ROOT, "training/tokenizers/en/tokenizer_spe_bpe_v128")
MODEL_FILE = os.path.join(EN_TOK_DIR, "tokenizer.model")
VOCAB_FILE = os.path.join(EN_TOK_DIR, "vocab.txt")

logging.basicConfig(level=logging.INFO)

def main():
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: {MODEL_FILE} not found.")
        print("   Did you run extract_english_tokenizer_v2.py?")
        return

    print(f"📖 Loading model: {MODEL_FILE}")
    
    # Load the SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(MODEL_FILE)
    
    vocab_size = sp.get_piece_size()
    print(f"   Vocab size: {vocab_size}")
    
    print(f"✍️  Generating {VOCAB_FILE}...")
    
    # Write vocab in the standard format NeMo expects
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        for i in range(vocab_size):
            piece = sp.id_to_piece(i)
            # NeMo doesn't strictly parse the score, but we'll mimic standard format
            # Format: token_string
            f.write(f"{piece}\n")
            
    print("✅ Success! Missing artifact created.")

if __name__ == "__main__":
    main()
