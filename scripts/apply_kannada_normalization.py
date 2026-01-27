"""
Apply Kannada Normalization and Measure WER Improvement
Reads predictions, normalizes both ground truth and predictions, calculates WER
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from kannada_normalization import KannadaNormalizer
import jiwer


def load_predictions(file_path: str) -> List[Dict]:
    """Load predictions from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_predictions(predictions: List[Dict], normalizer: KannadaNormalizer) -> List[Dict]:
    """Apply normalization to both ground_truth and prediction fields"""
    normalized = []
    
    for item in predictions:
        normalized_item = item.copy()
        
        if 'ground_truth' in item:
            normalized_item['ground_truth'] = normalizer.normalize(item['ground_truth'])
            normalized_item['ground_truth_original'] = item['ground_truth']
        
        if 'prediction' in item:
            normalized_item['prediction'] = normalizer.normalize(item['prediction'])
            normalized_item['prediction_original'] = item['prediction']
        
        normalized.append(normalized_item)
    
    return normalized


def calculate_wer(ground_truths: List[str], predictions: List[str]) -> Dict:
    """Calculate WER and related metrics"""
    
    # Calculate overall WER
    wer = jiwer.wer(ground_truths, predictions)
    
    # Calculate CER
    cer = jiwer.cer(ground_truths, predictions)
    
    # Get detailed measures
    measures = jiwer.compute_measures(ground_truths, predictions)
    
    return {
        'wer': wer * 100,  # Convert to percentage
        'cer': cer * 100,
        'substitutions': measures['substitutions'],
        'deletions': measures['deletions'],
        'insertions': measures['insertions'],
        'hits': measures['hits'],
        'total_words': measures['substitutions'] + measures['deletions'] + measures['hits'],
    }


def compare_wer(original_data: List[Dict], normalized_data: List[Dict]) -> Dict:
    """Compare WER before and after normalization"""
    
    # Extract texts for original
    orig_gt = [item['ground_truth'] for item in original_data]
    orig_pred = [item['prediction'] for item in original_data]
    
    # Extract texts for normalized
    norm_gt = [item['ground_truth'] for item in normalized_data]
    norm_pred = [item['prediction'] for item in normalized_data]
    
    # Calculate metrics
    original_metrics = calculate_wer(orig_gt, orig_pred)
    normalized_metrics = calculate_wer(norm_gt, norm_pred)
    
    # Calculate improvements
    wer_improvement = original_metrics['wer'] - normalized_metrics['wer']
    cer_improvement = original_metrics['cer'] - normalized_metrics['cer']
    
    return {
        'original': original_metrics,
        'normalized': normalized_metrics,
        'improvement': {
            'wer_absolute': wer_improvement,
            'wer_relative': (wer_improvement / original_metrics['wer']) * 100 if original_metrics['wer'] > 0 else 0,
            'cer_absolute': cer_improvement,
            'cer_relative': (cer_improvement / original_metrics['cer']) * 100 if original_metrics['cer'] > 0 else 0,
        }
    }


def find_changed_examples(original_data: List[Dict], normalized_data: List[Dict], max_examples: int = 20) -> List[Dict]:
    """Find examples where normalization changed the text"""
    changed = []
    
    for orig, norm in zip(original_data, normalized_data):
        gt_changed = orig['ground_truth'] != norm['ground_truth']
        pred_changed = orig['prediction'] != norm['prediction']
        
        if gt_changed or pred_changed:
            changed.append({
                'index': orig.get('index', '?'),
                'audio_filepath': orig.get('audio_filepath', ''),
                'ground_truth_orig': orig['ground_truth'],
                'ground_truth_norm': norm['ground_truth'],
                'prediction_orig': orig['prediction'],
                'prediction_norm': norm['prediction'],
                'gt_changed': gt_changed,
                'pred_changed': pred_changed,
            })
            
            if len(changed) >= max_examples:
                break
    
    return changed


def print_report(comparison: Dict, changed_examples: List[Dict]):
    """Print comprehensive report"""
    
    orig = comparison['original']
    norm = comparison['normalized']
    imp = comparison['improvement']
    
    print("=" * 80)
    print("KANNADA NORMALIZATION - WER IMPROVEMENT REPORT")
    print("=" * 80)
    print()
    
    print("BEFORE NORMALIZATION:")
    print("-" * 40)
    print(f"  WER:           {orig['wer']:.2f}%")
    print(f"  CER:           {orig['cer']:.2f}%")
    print(f"  Substitutions: {orig['substitutions']}")
    print(f"  Deletions:     {orig['deletions']}")
    print(f"  Insertions:    {orig['insertions']}")
    print(f"  Total Words:   {orig['total_words']}")
    print()
    
    print("AFTER NORMALIZATION:")
    print("-" * 40)
    print(f"  WER:           {norm['wer']:.2f}%")
    print(f"  CER:           {norm['cer']:.2f}%")
    print(f"  Substitutions: {norm['substitutions']}")
    print(f"  Deletions:     {norm['deletions']}")
    print(f"  Insertions:    {norm['insertions']}")
    print(f"  Total Words:   {norm['total_words']}")
    print()
    
    print("IMPROVEMENT:")
    print("-" * 40)
    print(f"  WER Absolute:  {imp['wer_absolute']:.2f}% {'↓' if imp['wer_absolute'] > 0 else '↑'}")
    print(f"  WER Relative:  {imp['wer_relative']:.2f}% {'improvement' if imp['wer_absolute'] > 0 else 'degradation'}")
    print(f"  CER Absolute:  {imp['cer_absolute']:.2f}% {'↓' if imp['cer_absolute'] > 0 else '↑'}")
    print(f"  CER Relative:  {imp['cer_relative']:.2f}% {'improvement' if imp['cer_absolute'] > 0 else 'degradation'}")
    print()
    
    if changed_examples:
        print("=" * 80)
        print(f"EXAMPLES OF NORMALIZATION CHANGES (Showing first {len(changed_examples)}):")
        print("=" * 80)
        
        for i, ex in enumerate(changed_examples[:10], 1):
            print(f"\nExample {i} (Index: {ex['index']}):")
            print("-" * 40)
            
            if ex['gt_changed']:
                print("  Ground Truth:")
                print(f"    Before: {ex['ground_truth_orig']}")
                print(f"    After:  {ex['ground_truth_norm']}")
            
            if ex['pred_changed']:
                print("  Prediction:")
                print(f"    Before: {ex['prediction_orig']}")
                print(f"    After:  {ex['prediction_norm']}")
    
    print()
    print("=" * 80)


def main():
    # File paths
    input_file = "/Users/chaitanyakartik/Downloads/predictions-2.json"
    output_file = "/Users/chaitanyakartik/Projects/asr-finetuning/docs/reports/model_v3_error_reports/predictions_normalized.json"
    
    print("Loading predictions...")
    original_data = load_predictions(input_file)
    print(f"Loaded {len(original_data)} predictions")
    
    print("\nApplying Kannada normalization...")
    normalizer = KannadaNormalizer()
    normalized_data = normalize_predictions(original_data, normalizer)
    
    print("\nCalculating WER improvements...")
    comparison = compare_wer(original_data, normalized_data)
    
    print("\nFinding changed examples...")
    changed_examples = find_changed_examples(original_data, normalized_data, max_examples=20)
    print(f"Found {len(changed_examples)} examples with changes")
    
    # Print report
    print_report(comparison, changed_examples)
    
    # Save normalized predictions
    print(f"\nSaving normalized predictions to:")
    print(f"  {output_file}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Done!")
    
    # Save comparison report
    report_file = output_path.parent / "normalization_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'comparison': comparison,
            'changed_examples_count': len(changed_examples),
            'changed_examples': changed_examples[:20]  # Save first 20
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Saved comparison report to: {report_file}")


if __name__ == "__main__":
    main()
