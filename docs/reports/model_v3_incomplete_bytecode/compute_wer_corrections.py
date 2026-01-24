#!/usr/bin/env python3
"""
Compute WER for bytecode corrections
Compares WER before and after fixing bytecode issues
"""

import json
from jiwer import wer, cer

# Read corrections file
with open('bytecode_corrections.json', 'r', encoding='utf-8') as f:
    corrections = json.load(f)

print("=" * 80)
print("WER/CER Analysis for Bytecode Corrections")
print("=" * 80)
print(f"Total entries with corrections: {len(corrections)}\n")

# Prepare data for metrics
ground_truths = [entry['ground_truth'] for entry in corrections]
predictions_original = [entry['prediction_original'].strip() for entry in corrections]
predictions_corrected = [entry['prediction_corrected'].strip() for entry in corrections]

# Compute metrics for original predictions
wer_original = wer(ground_truths, predictions_original) * 100
cer_original = cer(ground_truths, predictions_original) * 100

# Compute metrics for corrected predictions
wer_corrected = wer(ground_truths, predictions_corrected) * 100
cer_corrected = cer(ground_truths, predictions_corrected) * 100

# Display results
print("📊 ORIGINAL (with bytecode issues):")
print(f"   WER: {wer_original:.2f}%")
print(f"   CER: {cer_original:.2f}%")
print()

print("📊 CORRECTED (with proper Kannada characters):")
print(f"   WER: {wer_corrected:.2f}%")
print(f"   CER: {cer_corrected:.2f}%")
print()

# Calculate improvement
wer_improvement = wer_original - wer_corrected
cer_improvement = cer_original - cer_corrected

print("✨ IMPROVEMENT:")
print(f"   WER: {wer_improvement:+.2f}% (lower is better)")
print(f"   CER: {cer_improvement:+.2f}% (lower is better)")
print()

# Show some example corrections
print("=" * 80)
print("Sample Corrections (first 5):")
print("=" * 80)
for i, entry in enumerate(corrections[:5], 1):
    print(f"\n{i}. File: {entry['audio_filepath'].split('/')[-1]}")
    print(f"   Ground Truth: {entry['ground_truth'][:80]}...")
    print(f"   Original:     {entry['prediction_original'][:80]}...")
    print(f"   Corrected:    {entry['prediction_corrected'][:80]}...")

print("\n" + "=" * 80)
