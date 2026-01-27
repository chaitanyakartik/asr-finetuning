"""
Corpus-Based Kannada Normalizer
Uses vocabulary and rules learned from wiki corpus
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional


class CorpusBasedNormalizer:
    """Normalize Kannada text using corpus-derived patterns"""
    
    def __init__(self, vocab_path: str = None, rules_path: str = None):
        self.vocabulary = {}
        self.rules = []
        self.rule_map = {}  # Fast lookup
        
        if vocab_path:
            self.load_vocabulary(vocab_path)
        if rules_path:
            self.load_rules(rules_path)
    
    def load_vocabulary(self, path: str):
        """Load word vocabulary"""
        with open(path, 'r', encoding='utf-8') as f:
            self.vocabulary = json.load(f)
        print(f"✓ Loaded vocabulary: {len(self.vocabulary):,} words")
    
    def load_rules(self, path: str):
        """Load normalization rules"""
        with open(path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)
        
        # Build fast lookup map
        for rule in self.rules:
            self.rule_map[rule['source']] = rule['target']
        
        print(f"✓ Loaded normalization rules: {len(self.rules):,} rules")
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFC"""
        return unicodedata.normalize('NFC', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation"""
        # Various dash types
        text = text.replace('–', '-')
        text = text.replace('—', '-')
        text = text.replace('−', '-')
        
        # Quote marks
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        
        return text
    
    def normalize_numerals(self, text: str) -> str:
        """Convert Kannada numerals to ASCII"""
        kannada_to_ascii = str.maketrans('೦೧೨೩೪೫೬೭೮೯', '0123456789')
        return text.translate(kannada_to_ascii)
    
    def apply_rules(self, text: str) -> str:
        """Apply normalization rules"""
        # Apply each rule
        for source, target in self.rule_map.items():
            text = text.replace(source, target)
        
        return text
    
    def normalize_with_vocabulary(self, text: str) -> str:
        """
        Normalize using vocabulary preferences
        For compound words, prefer the form that exists in vocabulary
        """
        tokens = text.split()
        normalized_tokens = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check if this token + next token form a compound in vocab
            if i < len(tokens) - 1:
                next_token = tokens[i + 1]
                compound = token + next_token
                spaced = f"{token} {next_token}"
                
                # Get frequencies
                compound_freq = self.vocabulary.get(compound, 0)
                token_freq = self.vocabulary.get(token, 0)
                next_freq = self.vocabulary.get(next_token, 0)
                
                # If compound is more frequent, merge
                if compound_freq > min(token_freq, next_freq) * 2:
                    normalized_tokens.append(compound)
                    i += 2
                    continue
            
            # Check if token should be split
            if token in self.vocabulary:
                # Token exists as-is, keep it
                normalized_tokens.append(token)
            else:
                # Try splitting
                best_split = None
                best_score = 0
                
                for j in range(2, len(token) - 1):
                    part1 = token[:j]
                    part2 = token[j:]
                    
                    freq1 = self.vocabulary.get(part1, 0)
                    freq2 = self.vocabulary.get(part2, 0)
                    
                    if freq1 > 0 and freq2 > 0:
                        score = min(freq1, freq2)
                        if score > best_score:
                            best_score = score
                            best_split = (part1, part2)
                
                if best_split and best_score > 10:
                    # Good split found
                    normalized_tokens.extend(best_split)
                else:
                    # Keep original
                    normalized_tokens.append(token)
            
            i += 1
        
        return ' '.join(normalized_tokens)
    
    def normalize(self, text: str, use_vocab: bool = True, use_rules: bool = True) -> str:
        """
        Full normalization pipeline
        
        Args:
            text: Input text
            use_vocab: Use vocabulary-based normalization
            use_rules: Apply learned rules
        
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        # 1. Unicode normalization
        text = self.normalize_unicode(text)
        
        # 2. Punctuation
        text = self.normalize_punctuation(text)
        
        # 3. Numerals
        text = self.normalize_numerals(text)
        
        # 4. Apply rules
        if use_rules and self.rules:
            text = self.apply_rules(text)
        
        # 5. Vocabulary-based normalization
        if use_vocab and self.vocabulary:
            text = self.normalize_with_vocabulary(text)
        
        # 6. Final whitespace cleanup
        text = self.normalize_whitespace(text)
        
        return text


def create_normalizer(base_dir: str = None) -> CorpusBasedNormalizer:
    """Create normalizer with default paths"""
    if base_dir is None:
        base_dir = "/Users/chaitanyakartik/Projects/asr-finetuning/optimization/prediction_normalization"
    
    base_path = Path(base_dir)
    vocab_path = base_path / "vocabulary.json"
    rules_path = base_path / "normalization_rules.json"
    
    normalizer = CorpusBasedNormalizer()
    
    if vocab_path.exists():
        normalizer.load_vocabulary(str(vocab_path))
    else:
        print(f"⚠️  Vocabulary not found at {vocab_path}")
    
    if rules_path.exists():
        normalizer.load_rules(str(rules_path))
    else:
        print(f"⚠️  Rules not found at {rules_path}")
    
    return normalizer


# Example usage
if __name__ == "__main__":
    # Test normalizer
    normalizer = create_normalizer()
    
    test_texts = [
        "ಕೆಲವರ್ಷ ಹಲವುವೇಳೆ",
        "ಮನೆಮನೆಗೆ ತಲುಪಿಸುವ",
        "೧೨೩ ಎಂಟು",
    ]
    
    print("\nTest Normalization:")
    print("=" * 60)
    for text in test_texts:
        normalized = normalizer.normalize(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print()
