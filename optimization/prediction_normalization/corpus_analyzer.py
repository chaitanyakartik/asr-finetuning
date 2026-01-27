"""
Corpus-Based Kannada Text Analyzer
Builds word frequency dictionaries and compound word patterns from wiki corpus
"""

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
import unicodedata


class CorpusAnalyzer:
    """Analyze Kannada corpus to extract normalization patterns"""
    
    def __init__(self):
        self.word_freq = Counter()
        self.bigram_freq = Counter()
        self.trigram_freq = Counter()
        self.compound_candidates = defaultdict(set)
        
    def is_kannada_char(self, char: str) -> bool:
        """Check if character is Kannada"""
        return '\u0C80' <= char <= '\u0CFF'
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Kannada text into words"""
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Split on whitespace and punctuation, keep only Kannada words
        tokens = re.findall(r'[\u0C80-\u0CFF]+', text)
        return [t for t in tokens if len(t) > 0]
    
    def analyze_file(self, corpus_path: str, max_lines: int = None):
        """Analyze corpus file"""
        print(f"Analyzing corpus: {corpus_path}")
        
        line_count = 0
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_lines and line_count >= max_lines:
                    break
                
                tokens = self.tokenize(line.strip())
                
                # Count words
                self.word_freq.update(tokens)
                
                # Count bigrams
                if len(tokens) >= 2:
                    bigrams = zip(tokens[:-1], tokens[1:])
                    self.bigram_freq.update(bigrams)
                
                # Count trigrams
                if len(tokens) >= 3:
                    trigrams = zip(tokens[:-2], tokens[1:-1], tokens[2:])
                    self.trigram_freq.update(trigrams)
                
                line_count += 1
                
                if line_count % 100000 == 0:
                    print(f"  Processed {line_count:,} lines...")
        
        print(f"✓ Analyzed {line_count:,} lines")
        print(f"✓ Found {len(self.word_freq):,} unique words")
        print(f"✓ Found {len(self.bigram_freq):,} unique bigrams")
    
    def find_compound_variations(self, min_freq: int = 5) -> Dict[str, List[Tuple[str, int]]]:
        """
        Find compound word variations in corpus
        e.g., "ಕೆಲವರ್ಷ" vs "ಕೆಲವು ವರ್ಷ"
        """
        print("\nFinding compound word variations...")
        
        variations = defaultdict(list)
        
        # For each word, check if it could be a compound
        for word, freq in self.word_freq.most_common():
            if freq < min_freq:
                continue
            
            # Skip very short words
            if len(word) < 4:
                continue
            
            # Try splitting at different points
            for i in range(2, len(word) - 1):
                part1 = word[:i]
                part2 = word[i:]
                
                # Check if both parts exist as separate words
                if part1 in self.word_freq and part2 in self.word_freq:
                    # Check if the bigram exists
                    bigram_freq = self.bigram_freq.get((part1, part2), 0)
                    
                    if bigram_freq >= min_freq:
                        # Found a variation!
                        spaced_form = f"{part1} {part2}"
                        compound_freq = freq
                        spaced_freq = bigram_freq
                        
                        # Store both forms with frequencies
                        key = tuple(sorted([word, spaced_form]))
                        variations[key].append({
                            'compound': word,
                            'spaced': spaced_form,
                            'compound_freq': compound_freq,
                            'spaced_freq': spaced_freq,
                            'part1': part1,
                            'part2': part2,
                        })
        
        print(f"✓ Found {len(variations):,} compound variations")
        return dict(variations)
    
    def get_normalization_rules(self, variations: Dict) -> List[Dict]:
        """
        Convert variations to normalization rules
        Prefer the more frequent form
        """
        rules = []
        
        for key, var_list in variations.items():
            for var in var_list:
                # Prefer whichever form is more frequent
                if var['compound_freq'] > var['spaced_freq']:
                    # Compound form is more common
                    source = var['spaced']
                    target = var['compound']
                    freq_ratio = var['compound_freq'] / max(var['spaced_freq'], 1)
                else:
                    # Spaced form is more common
                    source = var['compound']
                    target = var['spaced']
                    freq_ratio = var['spaced_freq'] / max(var['compound_freq'], 1)
                
                # Only add if ratio is significant
                if freq_ratio > 1.5:
                    rules.append({
                        'source': source,
                        'target': target,
                        'source_freq': var['compound_freq'] if source == var['compound'] else var['spaced_freq'],
                        'target_freq': var['spaced_freq'] if target == var['spaced'] else var['compound_freq'],
                        'confidence': freq_ratio,
                    })
        
        # Sort by confidence
        rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        return rules
    
    def save_vocabulary(self, output_path: str, min_freq: int = 5):
        """Save word vocabulary with frequencies"""
        vocab = {
            word: freq 
            for word, freq in self.word_freq.most_common() 
            if freq >= min_freq
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved vocabulary ({len(vocab):,} words) to {output_path}")
    
    def save_normalization_rules(self, rules: List[Dict], output_path: str):
        """Save normalization rules"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved {len(rules):,} normalization rules to {output_path}")
    
    def generate_report(self) -> str:
        """Generate analysis report"""
        report = []
        report.append("=" * 80)
        report.append("KANNADA CORPUS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Total unique words: {len(self.word_freq):,}")
        report.append(f"Total word occurrences: {sum(self.word_freq.values()):,}")
        report.append(f"Total unique bigrams: {len(self.bigram_freq):,}")
        report.append("")
        report.append("TOP 50 MOST FREQUENT WORDS:")
        report.append("-" * 40)
        for word, freq in self.word_freq.most_common(50):
            report.append(f"  {word:20s} {freq:>10,}")
        report.append("")
        
        return "\n".join(report)


def main():
    corpus_path = "/Users/chaitanyakartik/Projects/asr-finetuning/data/training/wiki_corpus.txt"
    output_dir = Path("/Users/chaitanyakartik/Projects/asr-finetuning/optimization/prediction_normalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze corpus
    analyzer = CorpusAnalyzer()
    analyzer.analyze_file(corpus_path, max_lines=500000)  # Analyze first 500k lines for speed
    
    # Find variations
    variations = analyzer.find_compound_variations(min_freq=10)
    
    # Generate rules
    rules = analyzer.get_normalization_rules(variations)
    
    # Save outputs
    analyzer.save_vocabulary(output_dir / "vocabulary.json", min_freq=10)
    analyzer.save_normalization_rules(rules, output_dir / "normalization_rules.json")
    
    # Generate report
    report = analyzer.generate_report()
    print("\n" + report)
    
    with open(output_dir / "corpus_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ Analysis complete! Files saved to {output_dir}")


if __name__ == "__main__":
    main()
