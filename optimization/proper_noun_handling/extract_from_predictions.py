"""
Extract proper noun variants from YOUR PREDICTIONS (not corpus).
This is surgical - finds exactly what's hurting your WER.
"""

import json
from pathlib import Path
from collections import defaultdict


def levenshtein_distance(s1, s2, max_dist=2):
    """Fast Levenshtein with early termination."""
    if abs(len(s1) - len(s2)) > max_dist:
        return max_dist + 1
    
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        new_distances = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                new_distances.append(distances[i1])
            else:
                new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
        
        if min(new_distances) > max_dist:
            return max_dist + 1
        
        distances = new_distances
    
    return distances[-1]


def extract_mismatched_words(predictions):
    """
    Extract word-level mismatches from predictions.
    Returns pairs (ref_word, hyp_word) with frequency.
    """
    print("Extracting word-level mismatches...")
    
    mismatches = defaultdict(lambda: {'count': 0, 'examples': []})
    
    for item in predictions:
        ref_words = item['ground_truth'].split()
        hyp_words = item['prediction'].split()
        
        # Simple alignment: compare position-by-position
        # (this is rough but finds most mismatches)
        for i, (ref_w, hyp_w) in enumerate(zip(ref_words, hyp_words)):
            if ref_w != hyp_w:
                # Calculate edit distance
                dist = levenshtein_distance(ref_w, hyp_w, max_dist=3)
                
                # Only consider close variants (likely spelling differences)
                if 0 < dist <= 3:
                    key = f"{ref_w}|{hyp_w}"
                    mismatches[key]['count'] += 1
                    if len(mismatches[key]['examples']) < 3:
                        mismatches[key]['examples'].append({
                            'index': item.get('index', '?'),
                            'ref_sentence': item['ground_truth'],
                            'hyp_sentence': item['prediction']
                        })
    
    print(f"✓ Found {len(mismatches)} unique word mismatches\n")
    
    # Convert to list and sort by frequency
    mismatch_list = []
    for key, data in mismatches.items():
        ref_word, hyp_word = key.split('|')
        mismatch_list.append({
            'ref_word': ref_word,
            'hyp_word': hyp_word,
            'count': data['count'],
            'edit_distance': levenshtein_distance(ref_word, hyp_word, max_dist=3),
            'examples': data['examples']
        })
    
    mismatch_list.sort(key=lambda x: x['count'], reverse=True)
    
    return mismatch_list


def validate_against_corpus(mismatches, corpus_path, max_lines=500000):
    """
    Check which mismatches have both variants in corpus.
    If both appear, it's a legitimate spelling variant.
    """
    print(f"Validating against corpus ({max_lines:,} lines)...")
    
    # Get all words we need to check
    words_to_check = set()
    for m in mismatches:
        words_to_check.add(m['ref_word'])
        words_to_check.add(m['hyp_word'])
    
    print(f"  Checking {len(words_to_check)} unique words...")
    
    # Scan corpus
    word_exists = defaultdict(bool)
    line_count = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            if line_count > max_lines:
                break
            
            if line_count % 100000 == 0:
                print(f"    Scanned {line_count:,} lines...")
            
            for word in line.strip().split():
                if word in words_to_check:
                    word_exists[word] = True
    
    print(f"✓ Scanned corpus\n")
    
    # Tag mismatches with corpus validation
    for m in mismatches:
        ref_in_corpus = word_exists.get(m['ref_word'], False)
        hyp_in_corpus = word_exists.get(m['hyp_word'], False)
        
        m['ref_in_corpus'] = ref_in_corpus
        m['hyp_in_corpus'] = hyp_in_corpus
        m['both_in_corpus'] = ref_in_corpus and hyp_in_corpus
        
        # Recommendation
        if m['both_in_corpus']:
            m['recommendation'] = 'ACCEPT_BOTH'  # Legitimate variant
        elif not ref_in_corpus and hyp_in_corpus:
            m['recommendation'] = 'PREFER_HYP'  # Reference might be wrong
        elif ref_in_corpus and not hyp_in_corpus:
            m['recommendation'] = 'PREFER_REF'  # Model error
        else:
            m['recommendation'] = 'MANUAL_REVIEW'  # Neither in corpus
    
    return mismatches


def categorize_mismatches(mismatches):
    """Categorize by type of mismatch."""
    categorized = {
        'proper_nouns': [],
        'compound_spacing': [],
        'vowel_length': [],
        'consonant_variation': [],
        'other': []
    }
    
    for m in mismatches:
        ref, hyp = m['ref_word'], m['hyp_word']
        
        # Check patterns
        if ' ' in ref or ' ' in hyp:
            categorized['compound_spacing'].append(m)
        elif has_only_vowel_difference(ref, hyp):
            categorized['vowel_length'].append(m)
        elif has_proper_noun_pattern(ref) or has_proper_noun_pattern(hyp):
            categorized['proper_nouns'].append(m)
        elif has_consonant_variation(ref, hyp):
            categorized['consonant_variation'].append(m)
        else:
            categorized['other'].append(m)
    
    return categorized


def has_proper_noun_pattern(word):
    """Check if word looks like a proper noun."""
    # Place names
    place_indicators = ['ಪುರ', 'ಗಿ', 'ಗಿರಿ', 'ನಗರ', 'ಪಟ್ಟಣ', 'ನದಿ']
    for indicator in place_indicators:
        if word.endswith(indicator):
            return True
    
    # Capitalization-like patterns (not applicable to Kannada, but check length)
    # Proper nouns tend to be medium-length standalone words
    return False


def has_only_vowel_difference(w1, w2):
    """Check if difference is only vowel marks."""
    kannada_vowels = 'ಾಿೀುೂೃೄೆೇೈೊೋೌಂಃ'
    
    # Remove vowel marks and compare
    w1_no_vowels = ''.join(c for c in w1 if c not in kannada_vowels)
    w2_no_vowels = ''.join(c for c in w2 if c not in kannada_vowels)
    
    return w1_no_vowels == w2_no_vowels and w1 != w2


def has_consonant_variation(w1, w2):
    """Check for consonant variations (ದ/ದ್ದ, etc.)."""
    common_variations = [
        ('ದ', 'ದ್ದ'), ('ದ್ದ', 'ధ'), ('ವ', 'ವ್ವ'),
        ('ಗ', 'ಗ್ಗ'), ('ತ', 'ತ್ತ'), ('ರ', 'ರ್ರ')
    ]
    
    for v1, v2 in common_variations:
        if v1 in w1 and v2 in w2:
            return True
        if v2 in w1 and v1 in w2:
            return True
    
    return False


def main():
    # Paths
    predictions_file = Path("/Users/chaitanyakartik/Downloads/predictions.json")
    corpus_path = Path(__file__).parent.parent.parent / "data" / "training" / "wiki_corpus.txt"
    output_path = Path(__file__).parent / "mismatch_analysis.json"
    
    # Load predictions
    print(f"Loading predictions from: {predictions_file}")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    print(f"✓ Loaded {len(predictions)} predictions\n")
    
    # Extract mismatches
    mismatches = extract_mismatched_words(predictions)
    
    # Validate against corpus
    mismatches = validate_against_corpus(mismatches, str(corpus_path))
    
    # Categorize
    print("Categorizing mismatches...")
    categorized = categorize_mismatches(mismatches)
    
    print(f"\n{'='*80}")
    print("MISMATCH CATEGORIES")
    print(f"{'='*80}")
    print(f"  Proper nouns: {len(categorized['proper_nouns'])}")
    print(f"  Compound spacing: {len(categorized['compound_spacing'])}")
    print(f"  Vowel length: {len(categorized['vowel_length'])}")
    print(f"  Consonant variation: {len(categorized['consonant_variation'])}")
    print(f"  Other: {len(categorized['other'])}")
    
    # Show top proper noun mismatches
    print(f"\n{'='*80}")
    print("TOP 20 PROPER NOUN MISMATCHES")
    print(f"{'='*80}\n")
    
    proper_nouns = categorized['proper_nouns']
    for i, m in enumerate(proper_nouns[:20]):
        status = "✓" if m['both_in_corpus'] else "✗"
        print(f"{i+1}. {status} {m['ref_word']} → {m['hyp_word']} "
              f"(count={m['count']}, dist={m['edit_distance']}, {m['recommendation']})")
    
    # Save full analysis
    output_data = {
        'summary': {
            'total_mismatches': len(mismatches),
            'by_category': {k: len(v) for k, v in categorized.items()},
            'accept_both': sum(1 for m in mismatches if m['recommendation'] == 'ACCEPT_BOTH'),
            'prefer_ref': sum(1 for m in mismatches if m['recommendation'] == 'PREFER_REF'),
            'prefer_hyp': sum(1 for m in mismatches if m['recommendation'] == 'PREFER_HYP'),
            'manual_review': sum(1 for m in mismatches if m['recommendation'] == 'MANUAL_REVIEW')
        },
        'categorized': categorized,
        'all_mismatches': mismatches[:200]  # Top 200
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Full analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
