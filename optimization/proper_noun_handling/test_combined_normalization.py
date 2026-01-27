"""
Combined Normalizer: Proper Noun Variants + Corpus-Based Compound Splitting

Stacks two strategies:
1. Proper noun/variant normalization (validated from predictions)
2. Corpus-based compound splitting (from wiki analysis)
"""

import json
import unicodedata
from pathlib import Path


class CombinedNormalizer:
    def __init__(self, mismatch_analysis_path, vocab_path, rules_path):
        """Initialize with both normalization strategies."""
        
        # Load proper noun variants
        print("Loading proper noun variants...")
        with open(mismatch_analysis_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.variant_map = {}
        for category, mismatches in data['categorized'].items():
            for m in mismatches:
                if m['recommendation'] == 'ACCEPT_BOTH' and m['count'] >= 1:
                    ref, hyp = m['ref_word'], m['hyp_word']
                    # Bidirectional mapping to canonical form
                    canonical = min(ref, hyp)
                    self.variant_map[ref] = canonical
                    self.variant_map[hyp] = canonical
        
        print(f"  ✓ Loaded {len(self.variant_map)} variant mappings")
        
        # Load corpus-based rules
        print("Loading corpus-based rules...")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocabulary = json.load(f)
        
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)
            # Build map for quick lookup: "source" -> rule
            self.rule_map = {r['source']: r for r in self.rules}
        
        print(f"  ✓ Loaded {len(self.vocabulary):,} words, {len(self.rules)} rules")
    
    def normalize_unicode(self, text):
        """Unicode NFC normalization."""
        return unicodedata.normalize('NFC', text)
    
    def normalize_compounds(self, text):
        """Apply corpus-based compound splitting."""
        words = text.split()
        normalized = []
        
        for word in words:
            if word in self.rule_map:
                rule = self.rule_map[word]
                # Use the more common form
                if rule.get('prefer_split', False):
                    normalized.append(rule['split_form'])
                else:
                    normalized.append(word)
            else:
                normalized.append(word)
        
        return ' '.join(normalized)
    
    def normalize_with_vocabulary(self, text):
        """Prefer vocabulary forms over out-of-vocabulary."""
        words = text.split()
        normalized = []
        
        for word in words:
            if word in self.vocabulary:
                normalized.append(word)
            elif word in self.variant_map:
                # Use canonical variant
                normalized.append(self.variant_map[word])
            else:
                normalized.append(word)
        
        return ' '.join(normalized)
    
    def normalize_proper_nouns(self, text):
        """Apply proper noun variant normalization."""
        words = text.split()
        normalized = []
        
        for word in words:
            if word in self.variant_map:
                normalized.append(self.variant_map[word])
            else:
                normalized.append(word)
        
        return ' '.join(normalized)
    
    def normalize(self, text):
        """Apply combined normalization pipeline."""
        if not text:
            return text
        
        # Step 1: Unicode normalization
        text = self.normalize_unicode(text).strip()
        
        # Step 2: Proper noun variants (highest priority - validated)
        text = self.normalize_proper_nouns(text)
        
        # Step 3: Corpus-based compound splitting
        text = self.normalize_compounds(text)
        
        return text


def test_combined_normalizer(predictions_file, normalizer):
    """Test combined normalizer."""
    from jiwer import process_words, cer as compute_cer
    
    print(f"\n{'='*80}")
    print("TESTING COMBINED NORMALIZER")
    print(f"{'='*80}\n")
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Before
    refs_before = [p['ground_truth'] for p in predictions]
    hyps_before = [p['prediction'] for p in predictions]
    
    output_before = process_words(refs_before, hyps_before)
    wer_before = output_before.wer * 100
    cer_before = compute_cer(refs_before, hyps_before) * 100
    
    # After
    refs_after = [normalizer.normalize(p['ground_truth']) for p in predictions]
    hyps_after = [normalizer.normalize(p['prediction']) for p in predictions]
    
    output_after = process_words(refs_after, hyps_after)
    wer_after = output_after.wer * 100
    cer_after = compute_cer(refs_after, hyps_after) * 100
    
    print(f"\n📊 BEFORE:")
    print(f"  WER: {wer_before:.2f}% | CER: {cer_before:.2f}%")
    print(f"  S: {output_before.substitutions} | D: {output_before.deletions} | I: {output_before.insertions}")
    
    print(f"\n✨ AFTER (Combined):")
    print(f"  WER: {wer_after:.2f}% | CER: {cer_after:.2f}%")
    print(f"  S: {output_after.substitutions} | D: {output_after.deletions} | I: {output_after.insertions}")
    
    wer_diff = wer_before - wer_after
    print(f"\n🎯 IMPROVEMENT:")
    print(f"  WER: {wer_diff:+.2f}% absolute change")
    if wer_diff > 0:
        print(f"  ({(wer_diff/wer_before*100):.2f}% relative improvement)")
    
    print(f"\n{'='*80}\n")
    
    return {
        'wer_before': wer_before,
        'wer_after': wer_after,
        'improvement': wer_diff
    }


if __name__ == "__main__":
    # Paths
    mismatch_analysis = Path(__file__).parent / "mismatch_analysis.json"
    vocab_path = Path(__file__).parent.parent / "prediction_normalization" / "vocabulary.json"
    rules_path = Path(__file__).parent.parent / "prediction_normalization" / "normalization_rules.json"
    predictions_file = Path("/Users/chaitanyakartik/Downloads/predictions.json")
    
    # Check files exist
    for p in [mismatch_analysis, vocab_path, rules_path]:
        if not p.exists():
            print(f"❌ Missing: {p}")
            exit(1)
    
    # Initialize combined normalizer
    print("Initializing Combined Normalizer...")
    print("(Proper Noun Variants + Corpus-Based Compound Splitting)\n")
    
    normalizer = CombinedNormalizer(
        str(mismatch_analysis),
        str(vocab_path),
        str(rules_path)
    )
    
    # Test
    results = test_combined_normalizer(str(predictions_file), normalizer)
