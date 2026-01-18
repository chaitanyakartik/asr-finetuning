#!/usr/bin/env python3
"""
ASR Benchmark Runner (AI4Bharat Compatible)

Runs ASR model evaluation against versioned benchmark datasets.
Adapted for AI4Bharat IndicConformer models (Hybrid RNNT/CTC).
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from datetime import datetime

# NeMo imports
import nemo.collections.asr as nemo_asr

# Metrics imports
try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("⚠️  Warning: jiwer not installed. Install with: pip install jiwer")

def parse_args():
    parser = argparse.ArgumentParser(description="Run ASR benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Path to .nemo model file")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest json")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (Keep low for RNNT)")
    parser.add_argument("--decoder", type=str, default="rnnt", choices=["rnnt", "ctc"], help="Decoder type")
    parser.add_argument("--lang-id", type=str, default="kn", help="Language ID (e.g., 'kn', 'en', 'hi')")
    return parser.parse_args()

def run_benchmark(model, manifest_path, output_dir, batch_size, lang_id):
    """Run inference on a benchmark manifest"""
    print(f"   🚀 Running inference: {os.path.basename(manifest_path)}")
    
    audio_files = []
    ground_truths = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                audio_files.append(entry['audio_filepath'])
                ground_truths.append(entry.get('text', ""))
    
    print(f"      Files to transcribe: {len(audio_files)}")
    
    try:
        # --- FIX: Pass audio_files as POSITIONAL argument (no keyword) ---
        predictions = model.transcribe(
            audio_files,
            batch_size=batch_size,
            language_id=lang_id
        )
        
        # Handle tuple return (some versions return (texts, logits))
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Save detailed results
        predictions_path = os.path.join(output_dir, 'predictions.json')
        results = []
        for i, (audio, pred, truth) in enumerate(zip(audio_files, predictions, ground_truths)):
            results.append({
                'audio_filepath': audio,
                'ground_truth': truth,
                'prediction': pred,
                'index': i
            })
        
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"      ✅ Transcription complete")
        return {'status': 'completed', 'predictions_path': predictions_path}
    
    except Exception as e:
        print(f"      ❌ Transcription failed: {e}")
        return {'status': 'failed', 'error': str(e)}

def compute_metrics(predictions_path):
    """Compute WER/CER metrics"""
    if not JIWER_AVAILABLE: return {'wer': None, 'cer': None, 'status': 'error', 'error': 'jiwer missing'}
    try:
        with open(predictions_path, 'r', encoding='utf-8') as f: results = json.load(f)
        refs = [r['ground_truth'] for r in results]
        hyps = [r['prediction'] for r in results]
        
        w = wer(refs, hyps) * 100
        c = cer(refs, hyps) * 100
        
        print(f"      WER: {w:.2f}% | CER: {c:.2f}%")
        return {'wer': round(w, 2), 'cer': round(c, 2), 'status': 'completed'}
    except Exception as e: return {'status': 'failed', 'error': str(e)}

def generate_report(metrics, output_dir):
    """Generate and save the summary report"""
    report_path = os.path.join(output_dir, 'benchmark_report.json')
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'status': 'completed' if metrics.get('status') == 'completed' else 'failed'
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Report saved to: {report_path}")

def main():
    args = parse_args()
    
    print("=" * 80)
    print("ASR BENCHMARK RUNNER (AI4Bharat)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Decoder: {args.decoder} | Lang ID: {args.lang_id}")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. Load Model ---
    print("\n🔧 Loading ASR model...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nemo_asr.models.ASRModel.restore_from(restore_path=args.model)
        model.eval()
        model = model.to(device)
        
        # Set AI4Bharat specific decoder
        if hasattr(model, 'cur_decoder'):
            model.cur_decoder = args.decoder
            print(f"   ℹ️  Decoder set to: {model.cur_decoder}")
            
        print(f"   ✅ Model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return 1
    
    # --- 2. Run Benchmark ---
    benchmark_output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(benchmark_output_dir, exist_ok=True)
    
    res = run_benchmark(
        model,
        args.manifest,
        benchmark_output_dir,
        args.batch_size,
        args.lang_id
    )
    
    # --- 3. Compute Metrics & Generate Report ---
    metrics = {}
    if res['status'] == 'completed':
        metrics = compute_metrics(res['predictions_path'])
    else:
        metrics = {'status': 'failed', 'error': res.get('error')}
        
    generate_report(metrics, args.output_dir)
    
    print("\n✅ Benchmark run complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
