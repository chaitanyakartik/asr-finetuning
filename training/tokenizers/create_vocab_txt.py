#!/usr/bin/env python3
"""
Create vocab.txt from tokenizer.vocab
"""

vocab_file = "tokenizers/kn_master_v3000_retrained/tokenizer.vocab"
output_file = "tokenizers/kn_master_v3000_retrained/vocab.txt"

with open(vocab_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        # Extract just the token (first column before tab)
        token = line.split('\t')[0]
        f_out.write(token + '\n')

print(f"✅ Created {output_file}")
