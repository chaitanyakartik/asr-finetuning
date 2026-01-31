#!/usr/bin/env python3
"""
Fix bytecode issues in predictions JSON and generate report

Usage:
    python fix_bytecode_predictions.py --input predictions.json --output-corrected predictions_corrected.json --output-report report.json
"""

import json
import argparse
from pathlib import Path

try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("⚠️  Warning: jiwer not installed. WER/CER metrics will not be computed.")

# Bytecode to Kannada character mappings
BYTECODE_MAP = {
    '<0xE0><0xB2><0x94>': 'ಔ',  # Kannada letter AU
    '<0xE0><0xB2><0x8A>': 'ಊ',  # Kannada letter UU
    '<0xE0><0xB2><0x8E>': 'ಎ',  # Kannada letter E
    '<0xE0><0xB2><0x90>': 'ಐ',  # Kannada letter AI
    '<0xE0><0xB2><0xA2>': 'ಢ',  # Kannada letter DDHA
    '<0xE0><0xB2><0x9D>': 'ಝ',  # Kannada letter JHA
    '<0xE0><0xB2><0x8B>': 'ಋ',  # Kannada letter VOCALIC R
}


def fix_bytecodes(text):
    """Replace bytecodes with proper Kannada characters"""
    corrected = text
    for bytecode, kannada_char in BYTECODE_MAP.items():
        corrected = corrected.replace(bytecode, kannada_char)
    return corrected


def has_bytecodes(text):
    """Check if text contains any bytecodes"""
    return any(bytecode in text for bytecode in BYTECODE_MAP.keys())


def main():
    parser = argparse.ArgumentParser(description='Fix bytecode issues in predictions JSON')
    parser.add_argument('--input', required=True, help='Input predictions JSON file')
    parser.add_argument('--output-corrected', required=True, help='Output corrected predictions JSON file')
    parser.add_argument('--output-report', required=True, help='Output report JSON file')
    
    args = parser.parse_args()
    
    # Read input data
    print(f"📖 Reading: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process corrections
    corrections = []
    corrected_data = []
    
    for entry in data:
        original_prediction = entry.get('prediction', '')
        corrected_prediction = fix_bytecodes(original_prediction)
        
        # Create corrected entry
        corrected_entry = entry.copy()
        corrected_entry['prediction'] = corrected_prediction
        corrected_data.append(corrected_entry)
        
        # Track corrections
        if has_bytecodes(original_prediction):
            corrections.append({
                'index': entry.get('index', -1),
                'audio_filepath': entry.get('audio_filepath', ''),
                'ground_truth': entry.get('ground_truth', ''),
                'prediction_original': original_prediction,
                'prediction_corrected': corrected_prediction
            })
    
    # Save corrected predictions
    print(f"💾 Saving corrected predictions: {args.output_corrected}")
    with open(args.output_corrected, 'w', encoding='utf-8') as f:
        json.dump(corrected_data, f, indent=2, ensure_ascii=False)
    
    # Compute WER/CER metrics if jiwer is available
    metrics = {}
    if JIWER_AVAILABLE and len(data) > 0:
        print("📊 Computing WER/CER metrics...")
        ground_truths = [entry.get('ground_truth', '') for entry in data]
        predictions_original = [entry.get('prediction', '').strip() for entry in data]
        predictions_corrected_list = [entry.get('prediction', '').strip() for entry in corrected_data]
        
        # Filter out empty strings
        valid_indices = [i for i, gt in enumerate(ground_truths) if gt.strip()]
        ground_truths = [ground_truths[i] for i in valid_indices]
        predictions_original = [predictions_original[i] for i in valid_indices]
        predictions_corrected_list = [predictions_corrected_list[i] for i in valid_indices]
        
        if len(ground_truths) > 0:
            wer_original = wer(ground_truths, predictions_original) * 100
            cer_original = cer(ground_truths, predictions_original) * 100
            wer_corrected = wer(ground_truths, predictions_corrected_list) * 100
            cer_corrected = cer(ground_truths, predictions_corrected_list) * 100
            
            metrics = {
                'original': {
                    'wer': round(wer_original, 2),
                    'cer': round(cer_original, 2)
                },
                'corrected': {
                    'wer': round(wer_corrected, 2),
                    'cer': round(cer_corrected, 2)
                },
                'improvement': {
                    'wer_absolute': round(wer_original - wer_corrected, 2),
                    'cer_absolute': round(cer_original - cer_corrected, 2),
                    'wer_relative_percent': round((wer_original - wer_corrected) / wer_original * 100, 2) if wer_original > 0 else 0,
                    'cer_relative_percent': round((cer_original - cer_corrected) / cer_original * 100, 2) if cer_original > 0 else 0
                }
            }
    
    # Generate report
    report = {
        'total_samples': len(data),
        'corrections_made': len(corrections),
        'correction_rate': f"{len(corrections) / len(data) * 100:.2f}%",
        'bytecode_patterns_fixed': list(BYTECODE_MAP.keys()),
        'metrics': metrics if metrics else 'jiwer not available',
        'corrections_detail': corrections
    }
    
    # Save report
    print(f"📊 Saving report: {args.output_report}")
    with open(args.output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Total samples processed: {len(data)}")
    print(f"✅ Bytecode corrections made: {len(corrections)}")
    print(f"✅ Correction rate: {len(corrections) / len(data) * 100:.2f}%")
    
    if metrics:
        print(f"\n📊 METRICS:")
        print(f"   Original  - WER: {metrics['original']['wer']:.2f}%, CER: {metrics['original']['cer']:.2f}%")
        print(f"   Corrected - WER: {metrics['corrected']['wer']:.2f}%, CER: {metrics['corrected']['cer']:.2f}%")
        print(f"   Improvement:")
        print(f"     • WER: {metrics['improvement']['wer_absolute']:+.2f}% absolute ({metrics['improvement']['wer_relative_percent']:+.2f}% relative)")
        print(f"     • CER: {metrics['improvement']['cer_absolute']:+.2f}% absolute ({metrics['improvement']['cer_relative_percent']:+.2f}% relative)")
    
    print(f"\n✅ Output files created:")
    print(f"   - {args.output_corrected}")
    print(f"   - {args.output_report}")
    print("="*80)


if __name__ == '__main__':
    main()
