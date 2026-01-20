import os
import re
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIG ---
VOCAB_SIZE = 3000   # 3k is the "Sweet Spot" for 100M models
OUTPUT_DIR = "training/tokenizers/kn_master_v3000"
TEMP_TEXT_FILE = "kannada_corpus_dump.txt"

# OPTIONS: "wikipedia" or "indic_corp"
# wikipedia: Smaller, general knowledge, faster to download
# indic_corp: Massive, higher quality, more native Kannada variance
DATA_SOURCE = "wikipedia" 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_kannada_text(text):
    """
    Minimal cleaning:
    1. Remove non-Kannada characters (except English numbers/common punct if needed).
    2. Normalize whitespace.
    """
    if not text: return ""
    
    # 1. Remove URLs/HTML (Basic cleanup)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Keep Kannada range (0C80-0CFF), spaces, and basic punctuation
    # We also keep English numbers (0-9) as they appear frequently in Indian contexts
    text = re.sub(r'[^\u0C80-\u0CFF0-9\s\.\,\?\!]', ' ', text)
    
    # 3. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_tokenizer():
    print(f"⬇️  Loading Dataset: {DATA_SOURCE}...")
    
    # [Keep the dataset loading logic the same as before]
    if DATA_SOURCE == "wikipedia":
        try:
            dataset = load_dataset("wikimedia/wikipedia", "20231101.kn", split="train", trust_remote_code=True)
        except:
            dataset = load_dataset("wikimedia/wikipedia", "20231101.kn", split="train")
    elif DATA_SOURCE == "indic_corp":
        dataset = load_dataset("ai4bharat/IndicCorp", "kn", split="train", streaming=True)

    print("📝 Processing text (Splitting into sentences)...")
    
    count = 0
    # Stop at 1 Million sentences (plenty for 3k vocab)
    MAX_SENTENCES = 1000000 
    
    with open(TEMP_TEXT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(dataset):
            raw_text = item.get('text', item.get('sentence', ''))
            
            # --- FIX: Split Articles into Sentences ---
            # We split by common delimiters: . ? ! and newline
            # This prevents the "Too long line" error
            sentences = re.split(r'[.|?|!|\n]', raw_text)
            
            for sent in sentences:
                clean_line = clean_kannada_text(sent)
                
                # Filter: Keep lines that are 20-2000 chars long
                # (Too short = garbage, Too long = risky)
                if 20 < len(clean_line) < 2000: 
                    f.write(clean_line + "\n")
                    count += 1
            
            if count >= MAX_SENTENCES:
                break
    
    print(f"💾 Text dump saved. Extracted {count} valid sentences.")
    
    if count < 50000:
        print("⚠️ WARNING: Sentence count is still low. Tokenizer might be weak.")
    
    print(f"⚙️  Training SentencePiece Model (Vocab: {VOCAB_SIZE})...")
    
    # Added: max_sentence_length=10000 just to be safe
    spm.SentencePieceTrainer.train(
        input=TEMP_TEXT_FILE,
        model_prefix=os.path.join(OUTPUT_DIR, "tokenizer"),
        vocab_size=VOCAB_SIZE,
        character_coverage=0.9995,
        model_type='bpe',
        byte_fallback=True,
        input_sentence_size=MAX_SENTENCES,
        shuffle_input_sentence=True,
        max_sentence_length=10000  # <--- Increased limit
    )
    
    if os.path.exists(TEMP_TEXT_FILE):
        os.remove(TEMP_TEXT_FILE)
        
    print(f"✅ Tokenizer saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    build_tokenizer()
