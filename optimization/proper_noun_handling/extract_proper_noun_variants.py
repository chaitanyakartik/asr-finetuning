"""
Extract proper noun variants from Kannada Wikipedia corpus - OPTIMIZED VERSION

Uses:
1. Suffix-based bucketing (proper noun patterns)
2. Length-based sub-bucketing
3. Cheap pre-filters
4. Fast Levenshtein distance
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


def cheap_filter(w1, w2):
    """Fast pre-filter."""
    if abs(len(w1) - len(w2)) > 2:
        return False
    if w1[0] != w2[0]:
        return False
    if len(w1) > 3 and len(w2) > 3 and w1[-1] != w2[-1]:
        return False
    return True


def extract_proper_noun_variants(corpus_path, output_path, max_lines=500000):
    """Extract variants using optimized algorithm."""
    print(f"Reading corpus from: {corpus_path}")
    
    # Read and build frequency map
    word_freq = defaultdict(int)
    line_count = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            if line_count > max_lines:
                break
            if line_count % 100000 == 0:
                print(f"  Processed {line_count:,} lines...")
            
            for word in line.strip().split():
                word_freq[word] += 1
    
    print(f"✓ Processed {line_count:,} lines, {len(word_freq):,} unique words\n")
    
    # Filter to frequent words
    frequent_words = {w: f for w, f in word_freq.items() if f >= 3}
    print(f"Filtered to {len(frequent_words):,} words with freq >= 3\n")
    
    # Suffix-based bucketing
    print("Building suffix-based index...")
    suffix_patterns = ['ಪುರ', 'ಗಿ', 'ಗಿರಿ', 'ನಗರ', 'ಪಟ್ಟಣ', 'ಊರು', 'ಗ್ರಾಮ',
                      'ನದಿ', 'ಸಾಗರ', 'ಕೆರೆ', 'ಗುಡ್ಡ', 'ಬೆಟ್ಟ']
    
    suffix_buckets = defaultdict(list)
    for word in frequent_words:
        for suffix in suffix_patterns:
            if word.endswith(suffix) or suffix in word:
                suffix_buckets[suffix].append(word)
                break
    
    print(f"Created {len(suffix_buckets)} suffix buckets")
    for suffix, words in sorted(suffix_buckets.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"  {suffix}: {len(words)} words")
    print()
    
    # Find variants
    print("Finding variants...")
    variants = defaultdict(set)
    total_comparisons = 0
    found_variants = 0
    
    for suffix, word_list in suffix_buckets.items():
        if len(word_list) < 2:
            continue
        
        print(f"  Processing '{suffix}' ({len(word_list)} words)...")
        
        # Length bucketing within suffix
        length_buckets = defaultdict(list)
        for w in word_list:
            length_buckets[len(w)].append(w)
        
        for length, words in length_buckets.items():
            candidates = words + length_buckets.get(length - 1, []) + length_buckets.get(length + 1, [])
            
            for i, w1 in enumerate(candidates):
                for w2 in candidates[i+1:]:
                    total_comparisons += 1
                    
                    if not cheap_filter(w1, w2):
                        continue
                    
                    dist = levenshtein_distance(w1, w2, max_dist=2)
                    if 0 < dist <= 2:
                        variants[w1].add(w2)
                        variants[w2].add(w1)
                        found_variants += 1
        
        print(f"    Found {found_variants} variant pairs so far")
    
    print(f"\n✓ Total comparisons: {total_comparisons:,}")
    print(f"✓ Found {found_variants} variant pairs\n")
    
    # Categorize and save
    print("Categorizing variants...")
    
    categorized = {
        'place_names': {},
        'person_names': {},
        'other_proper_nouns': {},
        'statistics': {'total_variant_groups': 0, 'place_names': 0, 'person_names': 0, 'other': 0}
    }
    
    processed = set()
    for word, variant_set in variants.items():
        if word in processed:
            continue
        
        # Get connected component
        variant_group = {word}
        to_visit = variant_set.copy()
        while to_visit:
            current = to_visit.pop()
            if current not in variant_group:
                variant_group.add(current)
                to_visit.update(variants.get(current, set()) - variant_group)
        
        processed.update(variant_group)
        
        # Sort by frequency
        sorted_variants = sorted(variant_group, key=lambda w: word_freq[w], reverse=True)
        canonical = sorted_variants[0]
        alternatives = sorted_variants[1:]
        
        if not alternatives:
            continue
        
        # Categorize
        category = 'other'
        place_suffixes = ['ಪುರ', 'ಗಿ', 'ಗಿರಿ', 'ನಗರ', 'ಪಟ್ಟಣ', 'ಊರು']
        geo_keywords = ['ನದಿ', 'ಸಾಗರ', 'ಕೆರೆ', 'ಗುಡ್ಡ', 'ಬೆಟ್ಟ']
        
        for suffix in place_suffixes:
            if canonical.endswith(suffix):
                category = 'place_name'
                break
        
        if category == 'other':
            for keyword in geo_keywords:
                if keyword in canonical:
                    category = 'place_name'
                    break
        
        if category == 'other' and any(' ' in alt or ' ' in canonical for alt in alternatives):
            category = 'person_name'
        
        # Add to categorized
        entry = {
            'variants': alternatives,
            'frequency': word_freq[canonical],
            'variant_frequencies': {v: word_freq[v] for v in alternatives}
        }
        
        if category == 'place_name':
            categorized['place_names'][canonical] = entry
            categorized['statistics']['place_names'] += 1
        elif category == 'person_name':
            categorized['person_names'][canonical] = entry
            categorized['statistics']['person_names'] += 1
        else:
            categorized['other_proper_nouns'][canonical] = entry
            categorized['statistics']['other'] += 1
        
        categorized['statistics']['total_variant_groups'] += 1
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(categorized, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"\nStatistics:")
    print(f"  Place names: {categorized['statistics']['place_names']}")
    print(f"  Person names: {categorized['statistics']['person_names']}")
    print(f"  Other: {categorized['statistics']['other']}")
    print(f"  Total: {categorized['statistics']['total_variant_groups']}")


if __name__ == "__main__":
    corpus_path = Path(__file__).parent.parent.parent / "data" / "training" / "wiki_corpus.txt"
    output_path = Path(__file__).parent / "proper_noun_variants.json"
    
    if not corpus_path.exists():
        print(f"❌ Corpus not found: {corpus_path}")
        exit(1)
    
    extract_proper_noun_variants(str(corpus_path), str(output_path))
