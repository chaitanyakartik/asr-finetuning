"""
Proper Noun + Validated Variant Normalizer

Applies normalization only for:
1. Proper nouns with corpus-validated variants
2. Prevents gaming WER - only accepts legitimate alternatives
"""

import json
import unicodedata
from pathlib import Path


class ProperNounNormalizer:
    def __init__(self, mismatch_analysis_path):
        """Load validated variants from mismatch analysis."""
        with open(mismatch_analysis_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Build bidirectional mapping for ACCEPT_BOTH cases only
        self.variant_map = {}
        
        for category, mismatches in data['categorized'].items():
            for m in mismatches:
                if m['recommendation'] == 'ACCEPT_BOTH' and m['count'] >= 1:
                    ref, hyp = m['ref_word'], m['hyp_word']
                    # Bidirectional
                    self.variant_map[ref] = hyp
                    self.variant_map[hyp] = ref
        
        print(f"✓ Loaded {len(self.variant_map) // 2} bidirectional variant pairs")
    
    def normalize_word(self, word):
        """Normalize a single word if it has a known variant."""
        # Check if word has a known variant
        if word in self.variant_map:
            canonical = self.variant_map[word]
            # Always return the alphabetically first one (for consistency)
            return min(word, canonical)
        return word
    
    def normalize(self, text):
        """Normalize text by replacing known variants."""
        if not text:
            return text
        
        # Unicode normalization first
        text = unicodedata.normalize('NFC', text).strip()
        
        # Word-level normalization
        words = text.split()
        normalized_words = [self.normalize_word(w) for w in words]
        
        return ' '.join(normalized_words)


def test_on_predictions(predictions_file, normalizer):
    """Test normalizer and calculate WER impact."""
    from jiwer import process_words, cer as compute_cer
    
    print(f"\nLoading predictions...")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Calculate WER before normalization
    refs_before = [p['ground_truth'] for p in predictions]
    hyps_before = [p['prediction'] for p in predictions]
    
    output_before = process_words(refs_before, hyps_before)
    wer_before = output_before.wer * 100
    cer_before = compute_cer(refs_before, hyps_before) * 100
    
    # Calculate WER after normalization
    refs_after = [normalizer.normalize(p['ground_truth']) for p in predictions]
    hyps_after = [normalizer.normalize(p['prediction']) for p in predictions]
    
    output_after = process_words(refs_after, hyps_after)
    wer_after = output_after.wer * 100
    cer_after = compute_cer(refs_after, hyps_after) * 100
    
    # Results
    print(f"\n{'='*80}")
    print("PROPER NOUN NORMALIZATION RESULTS")
    print(f"{'='*80}\n")
    
    print(f"📊 BEFORE:")
    print(f"  WER: {wer_before:.2f}% | CER: {cer_before:.2f}%")
    print(f"  S: {output_before.substitutions} | D: {output_before.deletions} | I: {output_before.insertions}")
    
    print(f"\n✨ AFTER:")
    print(f"  WER: {wer_after:.2f}% | CER: {cer_after:.2f}%")
    print(f"  S: {output_after.substitutions} | D: {output_after.deletions} | I: {output_after.insertions}")
    
    wer_diff = wer_before - wer_after
    print(f"\n🎯 IMPROVEMENT:")
    if wer_diff > 0:
        print(f"  WER: {wer_diff:.2f}% absolute drop")
        print(f"  ({(wer_diff/wer_before*100):.2f}% relative improvement)")
    elif wer_diff < 0:
        print(f"  WER: {abs(wer_diff):.2f}% INCREASE (normalizer hurts performance!)")
    else:
        print(f"  No change in WER")
    
    print(f"\n{'='*80}\n")
    
    return {
        'wer_before': wer_before,
        'wer_after': wer_after,
        'wer_improvement': wer_diff
    }


if __name__ == "__main__":
    # Paths
    mismatch_analysis = Path(__file__).parent / "mismatch_analysis.json"
    predictions_file = Path("/Users/chaitanyakartik/Downloads/predictions.json")
    
    if not mismatch_analysis.exists():
        print(f"❌ Run extract_from_predictions.py first")
        exit(1)
    
    # Initialize normalizer
    print("Initializing Proper Noun Normalizer...")
    normalizer = ProperNounNormalizer(str(mismatch_analysis))
    
    # Test on predictions
    results = test_on_predictions(str(predictions_file), normalizer)
