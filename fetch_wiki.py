import os
import re
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIG ---
OUTPUT_FILE = "data/training/wiki_corpus_multilingual.txt" # New filename
DATA_SOURCE = "wikipedia" 
MAX_SENTENCES = 2000000 

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def clean_mixed_text(text):
    if not text: return ""
    
    # 1. Remove URLs/HTML
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # 2. THE FIX: Allow Kannada + English + Numbers + Basic Punctuation
    # Range: 
    #   \u0C80-\u0CFF : Kannada
    #   a-zA-Z        : English (Crucial for code-switching)
    #   0-9           : Numbers
    #   \.,\?!\'\-    : Punctuation (Comma, Dot, Question, etc.)
    text = re.sub(r'[^\u0C80-\u0CFFa-zA-Z0-9\s\.,\?!\'\-]', ' ', text)
    
    # 3. Collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_corpus():
    print(f"⬇️  Loading Dataset: {DATA_SOURCE}...")
    try:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.kn", split="train", trust_remote_code=True)
    except:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.kn", split="train")
    
    print(f"📝 Extracting Mixed Text (Kannada + English) to {OUTPUT_FILE}...")
    
    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(dataset):
            raw_text = item.get('text', item.get('sentence', ''))
            sentences = re.split(r'[.|?|!|\n]', raw_text)
            
            for sent in sentences:
                clean_line = clean_mixed_text(sent)
                if 20 < len(clean_line) < 2000: 
                    f.write(clean_line + "\n")
                    count += 1
                    
            if count >= MAX_SENTENCES:
                break
    
    print(f"✅ Success! Saved {count} sentences.")

if __name__ == "__main__":
    fetch_corpus()
