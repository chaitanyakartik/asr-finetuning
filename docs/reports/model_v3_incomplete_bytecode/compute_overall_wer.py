#!/usr/bin/env python3
"""
Compute overall WER/CER on full dataset
Compares metrics before and after bytecode corrections
"""

import json
from jiwer import wer, cer

# Read original data (with bytecode issues)
with open('benchmark_incomplete_v3.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# Read corrected data (with proper Kannada characters)
with open('benchmark_corrected_v3.json', 'r', encoding='utf-8') as f:
    corrected_data = json.load(f)

print("=" * 80)
print("OVERALL WER/CER Analysis - Full Dataset")
print("=" * 80)
print(f"Total samples: {len(original_data)}\n")

# Prepare data for metrics
ground_truths = [entry['ground_truth'] for entry in original_data]
predictions_original = [entry['prediction'].strip() for entry in original_data]
predictions_corrected = [entry['prediction'].strip() for entry in corrected_data]

# Count how many entries were actually changed
changed_count = sum(1 for orig, corr in zip(predictions_original, predictions_corrected) if orig != corr)
print(f"Entries with bytecode corrections: {changed_count}\n")

# Compute metrics for original predictions (with bytecode issues)
wer_original = wer(ground_truths, predictions_original) * 100
cer_original = cer(ground_truths, predictions_original) * 100

# Compute metrics for corrected predictions (with proper Kannada)
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

print("✨ OVERALL IMPROVEMENT:")
print(f"   WER: {wer_improvement:+.2f}% {'⬇️' if wer_improvement > 0 else '⬆️'}")
print(f"   CER: {cer_improvement:+.2f}% {'⬇️' if cer_improvement > 0 else '⬆️'}")
print()

# Percentage improvement
wer_pct_improvement = (wer_improvement / wer_original * 100) if wer_original > 0 else 0
cer_pct_improvement = (cer_improvement / cer_original * 100) if cer_original > 0 else 0

print("📈 RELATIVE IMPROVEMENT:")
print(f"   WER reduced by: {wer_pct_improvement:.2f}%")
print(f"   CER reduced by: {cer_pct_improvement:.2f}%")
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Fixing {changed_count} bytecode issues improved:")
print(f"  • Word Error Rate from {wer_original:.2f}% → {wer_corrected:.2f}%")
print(f"  • Character Error Rate from {cer_original:.2f}% → {cer_corrected:.2f}%")
print("=" * 80)
