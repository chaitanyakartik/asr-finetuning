"""
Compare three normalization strategies:
1. Conservative: Only Unicode + whitespace
2. Aggressive: Corpus-based compound splitting + vocabulary matching
3. Style Guide: Prescriptive orthographic rules
"""

import json
import sys
from pathlib import Path
from jiwer import process_words, cer as compute_cer

from corpus_normalizer import CorpusBasedNormalizer
from corpus_normalizer_conservative import ConservativeKannadaNormalizer
from orthographic_style_guide import OrthographicStyleNormalizer


def load_predictions(file_path: str):
    """Load predictions from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_normalization(predictions, normalizer, strategy_name):
    """Apply normalization and calculate WER."""
    print(f"\n{'='*80}")
    print(f"Testing: {strategy_name}")
    print(f"{'='*80}\n")
    
    normalized_predictions = []
    references = []
    hypotheses = []
    
    for item in predictions:
        ground_truth = item['ground_truth']
        prediction = item['prediction']
        
        # Normalize both
        norm_gt = normalizer.normalize(ground_truth)
        norm_pred = normalizer.normalize(prediction)
        
        references.append(norm_gt)
        hypotheses.append(norm_pred)
        
        normalized_predictions.append({
            **item,
            'ground_truth_normalized': norm_gt,
            'prediction_normalized': norm_pred,
            'ground_truth_original': ground_truth,
            'prediction_original': prediction
        })
    
    # Calculate WER
    output = process_words(references, hypotheses)
    wer = output.wer * 100
    
    # Calculate CER separately
    cer = compute_cer(references, hypotheses) * 100
    
    print(f"📊 Results:")
    print(f"  WER: {wer:.2f}%")
    print(f"  CER: {cer:.2f}%")
    print(f"  Substitutions: {output.substitutions}")
    print(f"  Deletions: {output.deletions}")
    print(f"  Insertions: {output.insertions}")
    
    return {
        'strategy': strategy_name,
        'wer': wer,
        'cer': cer,
        'substitutions': output.substitutions,
        'deletions': output.deletions,
        'insertions': output.insertions,
        'normalized_predictions': normalized_predictions
    }


def show_examples(results_dict, num_examples=10):
    """Show example differences between strategies."""
    print(f"\n{'='*80}")
    print(f"Example Differences (first {num_examples})")
    print(f"{'='*80}\n")
    
    cons_preds = results_dict['conservative']['normalized_predictions']
    agg_preds = results_dict['aggressive']['normalized_predictions']
    style_preds = results_dict['style_guide']['normalized_predictions']
    
    shown = 0
    for i, (cons, agg, style) in enumerate(zip(cons_preds, agg_preds, style_preds)):
        cons_pred = cons['prediction_normalized']
        agg_pred = agg['prediction_normalized']
        style_pred = style['prediction_normalized']
        
        # Show if any differ
        if len({cons_pred, agg_pred, style_pred}) > 1 and shown < num_examples:
            print(f"Example {shown + 1} (index {i}):")
            print(f"  Original:     {cons['prediction_original']}")
            print(f"  Conservative: {cons_pred}")
            print(f"  Aggressive:   {agg_pred}")
            print(f"  Style Guide:  {style_pred}")
            print(f"  Ground truth: {cons['ground_truth_normalized']}")
            print()
            shown += 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_normalization_strategies_v2.py <predictions.json>")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    
    print(f"Loading predictions from: {predictions_file}")
    predictions = load_predictions(predictions_file)
    print(f"✓ Loaded {len(predictions)} predictions\n")
    
    # Strategy 1: Conservative
    print("Initializing Conservative Normalizer...")
    conservative_normalizer = ConservativeKannadaNormalizer()
    conservative_result = apply_normalization(
        predictions, 
        conservative_normalizer, 
        "Conservative (Unicode + Whitespace only)"
    )
    
    # Strategy 2: Aggressive (corpus-based)
    print("\nInitializing Aggressive Normalizer...")
    vocab_path = Path(__file__).parent / "vocabulary.json"
    rules_path = Path(__file__).parent / "normalization_rules.json"
    
    if not vocab_path.exists() or not rules_path.exists():
        print("⚠️  Corpus analysis files not found. Run corpus_analyzer.py first.")
        sys.exit(1)
    
    aggressive_normalizer = CorpusBasedNormalizer(
        str(vocab_path),
        str(rules_path)
    )
    aggressive_result = apply_normalization(
        predictions, 
        aggressive_normalizer, 
        "Aggressive (Corpus-based compound splitting)"
    )
    
    # Strategy 3: Style Guide (prescriptive rules)
    print("\nInitializing Style Guide Normalizer...")
    style_normalizer = OrthographicStyleNormalizer()
    style_result = apply_normalization(
        predictions,
        style_normalizer,
        "Style Guide (Prescriptive orthographic rules)"
    )
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    cons_wer = conservative_result['wer']
    agg_wer = aggressive_result['wer']
    style_wer = style_result['wer']
    
    print(f"Strategy                    | WER      | CER      | S     | D    | I")
    print(f"{'-'*80}")
    print(f"Conservative (baseline)     | {cons_wer:6.2f}%  | {conservative_result['cer']:6.2f}%  | "
          f"{conservative_result['substitutions']:5d} | {conservative_result['deletions']:4d} | "
          f"{conservative_result['insertions']:4d}")
    print(f"Aggressive (corpus-based)   | {agg_wer:6.2f}%  | {aggressive_result['cer']:6.2f}%  | "
          f"{aggressive_result['substitutions']:5d} | {aggressive_result['deletions']:4d} | "
          f"{aggressive_result['insertions']:4d}")
    print(f"Style Guide (prescriptive)  | {style_wer:6.2f}%  | {style_result['cer']:6.2f}%  | "
          f"{style_result['substitutions']:5d} | {style_result['deletions']:4d} | "
          f"{style_result['insertions']:4d}")
    print(f"{'-'*80}")
    
    # Find best strategy
    results = [
        ('Conservative', cons_wer),
        ('Aggressive', agg_wer),
        ('Style Guide', style_wer)
    ]
    best_strategy, best_wer = min(results, key=lambda x: x[1])
    
    print(f"\n🏆 WINNER: {best_strategy} ({best_wer:.2f}% WER)")
    print(f"\nComparisons vs Conservative baseline:")
    print(f"  • Aggressive: {cons_wer - agg_wer:+.2f}% WER change")
    print(f"  • Style Guide: {cons_wer - style_wer:+.2f}% WER change")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}\n")
    
    agg_diff = cons_wer - agg_wer
    style_diff = cons_wer - style_wer
    
    print(f"Conservative → Aggressive: {agg_diff:+.2f}%")
    if agg_diff > 0.5:
        print("  ✨ Corpus-based splitting HELPS significantly")
        print("  ⚠️  But may overfit to Wikipedia style conventions")
    elif agg_diff < -0.5:
        print("  ❌ Corpus-based splitting HURTS performance")
    else:
        print("  ⚖️  Minimal impact from corpus-based approach")
    
    print(f"\nConservative → Style Guide: {style_diff:+.2f}%")
    if style_diff > 0.5:
        print("  ✨ Prescriptive rules HELP significantly")
        print("  ✓  Fusing compounds matches ground truth conventions")
    elif style_diff < -0.5:
        print("  ❌ Prescriptive rules HURT performance")
        print("  ⚠️  Ground truth may use different conventions")
    else:
        print("  ⚖️  Minimal impact from style guide rules")
    
    print(f"\n📋 RECOMMENDATION:")
    if best_strategy == 'Conservative':
        print("  Use CONSERVATIVE: Keep it simple, minimal normalization")
    elif best_strategy == 'Aggressive':
        print("  Use AGGRESSIVE: Corpus conventions match your ground truth")
        print("  Document: 'Normalized to Kannada Wikipedia style'")
    else:
        print("  Use STYLE GUIDE: Prescriptive rules match your ground truth")
        print("  Document: 'Normalized per orthographic style guide'")
    
    # Show examples
    results_dict = {
        'conservative': conservative_result,
        'aggressive': aggressive_result,
        'style_guide': style_result
    }
    show_examples(results_dict, num_examples=15)
    
    # Save results
    output_file = Path(predictions_file).parent / "normalization_strategy_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'conservative': {
                'wer': conservative_result['wer'],
                'cer': conservative_result['cer'],
                'substitutions': conservative_result['substitutions'],
                'deletions': conservative_result['deletions'],
                'insertions': conservative_result['insertions'],
            },
            'aggressive': {
                'wer': aggressive_result['wer'],
                'cer': aggressive_result['cer'],
                'substitutions': aggressive_result['substitutions'],
                'deletions': aggressive_result['deletions'],
                'insertions': aggressive_result['insertions'],
            },
            'style_guide': {
                'wer': style_result['wer'],
                'cer': style_result['cer'],
                'substitutions': style_result['substitutions'],
                'deletions': style_result['deletions'],
                'insertions': style_result['insertions'],
            },
            'best_strategy': best_strategy.lower().replace(' ', '_'),
            'wer_improvements': {
                'aggressive_vs_conservative': agg_diff,
                'style_guide_vs_conservative': style_diff
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed comparison saved to: {output_file}")


if __name__ == "__main__":
    main()
