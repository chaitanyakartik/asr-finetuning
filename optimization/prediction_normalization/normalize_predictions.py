"""
Apply Normalization to Any Prediction File
Configurable script that won't overwrite existing reports
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from corpus_normalizer import create_normalizer
import jiwer


def load_predictions(file_path: str):
    """Load predictions JSON"""
    print(f"Loading predictions from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} predictions")
    return data


def normalize_predictions(data, normalizer):
    """Apply normalization"""
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


def calculate_wer(ground_truths, predictions):
    """Calculate WER metrics"""
    wer = jiwer.wer(ground_truths, predictions)
    cer = jiwer.cer(ground_truths, predictions)
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


def compare_wer(original, normalized):
    """Compare metrics"""
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


def print_report(comparison, input_file, changes_count):
    """Print report"""
    before = comparison['before']
    after = comparison['after']
    imp = comparison['improvement']
    
    print("\n" + "=" * 80)
    print(f"🔥 NORMALIZATION RESULTS: {Path(input_file).name}")
    print("=" * 80)
    print()
    
    print("📊 BEFORE:")
    print(f"  WER: {before['wer']:.2f}% | CER: {before['cer']:.2f}%")
    print(f"  S: {before['substitutions']:,} | D: {before['deletions']:,} | I: {before['insertions']:,}")
    print()
    
    print("✨ AFTER:")
    print(f"  WER: {after['wer']:.2f}% | CER: {after['cer']:.2f}%")
    print(f"  S: {after['substitutions']:,} | D: {after['deletions']:,} | I: {after['insertions']:,}")
    print()
    
    wer_symbol = "📉" if imp['wer_absolute_drop'] > 0 else "📈"
    print("🎯 IMPROVEMENT:")
    print(f"  {wer_symbol} WER: {imp['wer_absolute_drop']:.2f}% absolute drop")
    print(f"     ({imp['wer_relative']:.2f}% relative improvement)")
    print(f"  Changes: {changes_count} examples modified")
    print("=" * 80)


def main(input_file: str, output_prefix: str = None):
    """
    Main processing function
    
    Args:
        input_file: Path to predictions JSON
        output_prefix: Optional prefix for output files (default: timestamp)
    """
    # Generate unique output prefix if not provided
    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"run_{timestamp}"
    
    # Setup paths
    base_dir = Path("/Users/chaitanyakartik/Projects/asr-finetuning/optimization/prediction_normalization")
    input_name = Path(input_file).stem
    
    output_predictions = base_dir / f"{output_prefix}_{input_name}_normalized.json"
    output_report = base_dir / f"{output_prefix}_{input_name}_report.json"
    
    # Load normalizer
    print("\n📚 Loading corpus-based normalizer...")
    normalizer = create_normalizer(str(base_dir))
    
    # Load and process
    original_data = load_predictions(input_file)
    normalized_data = normalize_predictions(original_data, normalizer)
    
    # Compare
    print("\n📊 Calculating metrics...")
    comparison = compare_wer(original_data, normalized_data)
    
    # Count changes
    changes_count = sum(
        1 for orig, norm in zip(original_data, normalized_data)
        if orig['ground_truth'] != norm['ground_truth'] or 
           orig['prediction'] != norm['prediction']
    )
    
    # Print report
    print_report(comparison, input_file, changes_count)
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    print(f"  Predictions: {output_predictions.name}")
    print(f"  Report: {output_report.name}")
    
    with open(output_predictions, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)
    
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump({
            'input_file': str(input_file),
            'timestamp': datetime.now().isoformat(),
            'comparison': comparison,
            'changes_count': changes_count,
        }, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Done!")
    return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply Kannada normalization to predictions')
    parser.add_argument('input_file', help='Path to predictions JSON file')
    parser.add_argument('--prefix', help='Output file prefix (default: timestamp)', default=None)
    
    args = parser.parse_args()
    
    main(args.input_file, args.prefix)
