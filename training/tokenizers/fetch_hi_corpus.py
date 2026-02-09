from datasets import load_dataset

def download_hindi_subset():
    print("Streaming Hindi data from IndicCorpV2...")
    
    # We point directly to the data files in the repo
    # This avoids the 'config not found' error
    data_files = {"train": "data/hi-1.txt"} 
    
    # streaming=True ensures we don't download the 26GB file
    dataset = load_dataset("ai4bharat/IndicCorpV2", data_files=data_files, split="train", streaming=True)
    
    output_file = "hindi_corpus_for_tokenizer.txt"
    # For a 1.5k vocab tokenizer, 1-2 million lines is plenty of context
    max_lines = 1000000 
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            # In raw text datasets, the line is usually under the key 'text'
            line = entry["text"].strip()
            if line:
                f.write(line + "\n")
                count += 1
            
            if count % 100000 == 0:
                print(f"Captured {count} lines...")
                
            if count >= max_lines:
                break

    print(f"Success! Saved {count} lines to {output_file}")

if __name__ == "__main__":
    download_hindi_subset()