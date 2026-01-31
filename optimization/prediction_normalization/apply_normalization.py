"""
Apply Corpus-Based Normalization and Measure WER Improvement
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from corpus_normalizer import create_normalizer
import jiwer


def load_predictions(file_path: str) -> List[Dict]:
    """Load predictions JSON"""
    print(f"Loading predictions from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} predictions")
    return data


def normalize_predictions(data: List[Dict], normalizer) -> List[Dict]:
    """Apply normalization to predictions"""
    print("\nNormalizing predictions...")
    normalized = []
    
    for item in data:
        norm_item = item.copy()
        
        if 'ground_truth' in item:
            norm_item['ground_truth_original'] = item['ground_truth']
            norm_item['ground_truth'] = normalizer.normalize(item['ground_truth'])
        
        if 'prediction' in item:
            norm_item['prediction_original'] = item['prediction']
            norm_item['prediction'] = normalizer.normalize(item['prediction'])
        
        normalized.append(norm_item)
    
    print(f"✓ Normalized {len(normalized)} predictions")
    return normalized


def calculate_wer(ground_truths: List[str], predictions: List[str]) -> Dict:
    """Calculate WER metrics"""
    wer = jiwer.wer(ground_truths, predictions)
    cer = jiwer.cer(ground_truths, predictions)
    
    # Get detailed word counts
    output = jiwer.process_words(ground_truths, predictions)
    
    return {
        'wer': wer * 100,
        'cer': cer * 100,
        'substitutions': output.substitutions,
        'deletions': output.deletions,
        'insertions': output.insertions,
        'hits': output.hits,
        'total_words': output.substitutions + output.deletions + output.hits,
    }


def compare_wer(original: List[Dict], normalized: List[Dict]) -> Dict:
    """Compare WER before and after"""
    orig_gt = [item['ground_truth'] for item in original]
    orig_pred = [item['prediction'] for item in original]
    norm_gt = [item['ground_truth'] for item in normalized]
    norm_pred = [item['prediction'] for item in normalized]
    
    original_metrics = calculate_wer(orig_gt, orig_pred)
    normalized_metrics = calculate_wer(norm_gt, norm_pred)
    
    wer_drop = original_metrics['wer'] - normalized_metrics['wer']
    cer_drop = original_metrics['cer'] - normalized_metrics['cer']
    
    return {
        'before': original_metrics,
        'after': normalized_metrics,
        'improvement': {
            'wer_absolute_drop': wer_drop,
            'wer_relative': (wer_drop / original_metrics['wer'] * 100) if original_metrics['wer'] > 0 else 0,
            'cer_absolute_drop': cer_drop,
            'cer_relative': (cer_drop / original_metrics['cer'] * 100) if original_metrics['cer'] > 0 else 0,
        }
    }


def find_changes(original: List[Dict], normalized: List[Dict], n: int = 30) -> List[Dict]:
    """Find examples where normalization changed text"""
    changes = []
    
    for orig, norm in zip(original, normalized):
        gt_diff = orig['ground_truth'] != norm['ground_truth']
        pred_diff = orig['prediction'] != norm['prediction']
        
        if gt_diff or pred_diff:
            changes.append({
                'index': orig.get('index', '?'),
                'file': orig.get('audio_filepath', ''),
                'gt_before': orig['ground_truth'],
                'gt_after': norm['ground_truth'],
                'pred_before': orig['prediction'],
                'pred_after': norm['prediction'],
                'gt_changed': gt_diff,
                'pred_changed': pred_diff,
            })
            
            if len(changes) >= n:
                break
    
    return changes


def print_report(comparison: Dict, changes: List[Dict]):
    """Print detailed report"""
    before = comparison['before']
    after = comparison['after']
    imp = comparison['improvement']
    
    print("\n" + "=" * 80)
    print("🔥 CORPUS-BASED NORMALIZATION - WER IMPROVEMENT REPORT")
    print("=" * 80)
    print()
    
    print("📊 BEFORE NORMALIZATION:")
    print("-" * 40)
    print(f"  WER:           {before['wer']:.2f}%")
    print(f"  CER:           {before['cer']:.2f}%")
    print(f"  Substitutions: {before['substitutions']:,}")
    print(f"  Deletions:     {before['deletions']:,}")
    print(f"  Insertions:    {before['insertions']:,}")
    print(f"  Total Words:   {before['total_words']:,}")
    print()
    
    print("✨ AFTER NORMALIZATION:")
    print("-" * 40)
    print(f"  WER:           {after['wer']:.2f}%")
    print(f"  CER:           {after['cer']:.2f}%")
    print(f"  Substitutions: {after['substitutions']:,}")
    print(f"  Deletions:     {after['deletions']:,}")
    print(f"  Insertions:    {after['insertions']:,}")
    print(f"  Total Words:   {after['total_words']:,}")
    print()
    
    print("🎯 IMPROVEMENT:")
    print("-" * 40)
    wer_symbol = "📉" if imp['wer_absolute_drop'] > 0 else "📈"
    cer_symbol = "📉" if imp['cer_absolute_drop'] > 0 else "📈"
    
    print(f"  {wer_symbol} WER Drop:      {imp['wer_absolute_drop']:.2f}% absolute")
    print(f"     ({imp['wer_relative']:.2f}% relative improvement)")
    print(f"  {cer_symbol} CER Drop:      {imp['cer_absolute_drop']:.2f}% absolute")
    print(f"     ({imp['cer_relative']:.2f}% relative improvement)")
    print()
    
    if changes:
        print("=" * 80)
        print(f"📝 NORMALIZATION EXAMPLES (First {min(10, len(changes))}):")
        print("=" * 80)
        
        for i, change in enumerate(changes[:10], 1):
            print(f"\n[Example {i}] Index: {change['index']}")
            print("-" * 40)
            
            if change['gt_changed']:
                print("  Ground Truth:")
                print(f"    ❌ Before: {change['gt_before']}")
                print(f"    ✅ After:  {change['gt_after']}")
            
            if change['pred_changed']:
                print("  Prediction:")
                print(f"    ❌ Before: {change['pred_before']}")
                print(f"    ✅ After:  {change['pred_after']}")
    
    print("\n" + "=" * 80)


def main():
    base_dir = Path("/Users/chaitanyakartik/Projects/asr-finetuning/optimization/prediction_normalization")
    
    # Input/output paths
    input_file = "/Users/chaitanyakartik/Downloads/predictions-2.json"
    output_file = base_dir / "predictions_normalized.json"
    report_file = base_dir / "wer_improvement_report.json"
    
    # Step 1: Load data
    original_data = load_predictions(input_file)
    
    # Step 2: Create normalizer
    print("\n📚 Loading corpus-based normalizer...")
    normalizer = create_normalizer(str(base_dir))
    
    # Step 3: Normalize
    normalized_data = normalize_predictions(original_data, normalizer)
    
    # Step 4: Compare WER
    print("\n📊 Calculating WER metrics...")
    comparison = compare_wer(original_data, normalized_data)
    
    # Step 5: Find changes
    print("\n🔍 Finding normalization changes...")
    changes = find_changes(original_data, normalized_data, n=50)
    print(f"✓ Found {len(changes)} examples with changes")
    
    # Step 6: Print report
    print_report(comparison, changes)
    
    # Step 7: Save outputs
    print(f"\n💾 Saving outputs...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved normalized predictions: {output_file}")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'comparison': comparison,
            'total_changes': len(changes),
            'example_changes': changes[:30]
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved report: {report_file}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
