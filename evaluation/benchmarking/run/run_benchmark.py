#!/usr/bin/env python3
"""
ASR Benchmark Runner

Runs ASR model evaluation against versioned benchmark datasets.

Usage:
    python run_benchmark.py --model path/to/model.nemo --benchmark-set v1 --output-dir ../reports/run_001
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# NeMo imports
import nemo.collections.asr as nemo_asr

# Metrics imports
try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("⚠️  Warning: jiwer not installed. Install with: pip install jiwer")

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description="Run ASR benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .nemo model file or pretrained model name"
    )
    parser.add_argument(
        "--benchmark-set",
        type=str,
        default="v1",
        help="Benchmark version to run (default: v1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Specific benchmarks to run (default: all available)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)"
    )
    return parser.parse_args()

def discover_benchmarks(benchmark_dir, benchmark_set):
    """Discover all available benchmark manifests in the specified version"""
    version_dir = os.path.join(benchmark_dir, benchmark_set)
    
    if not os.path.exists(version_dir):
        print(f"❌ Benchmark set '{benchmark_set}' not found at {version_dir}")
        return []
    
    benchmarks = []
    for f in os.listdir(version_dir):
        if f.endswith('.json'):
            benchmark_name = f.replace('.json', '')
            manifest_path = os.path.join(version_dir, f)
            benchmarks.append({
                'name': benchmark_name,
                'manifest': manifest_path
            })
    
    return benchmarks

def run_curation_pipeline(pipeline_name):
    """Run a benchmark curation pipeline"""
    curation_dir = Path(__file__).parent.parent / "curation"
    pipeline_path = curation_dir / f"{pipeline_name}.py"
    
    if not pipeline_path.exists():
        print(f"⚠️  Curation pipeline not found: {pipeline_path}")
        return False
    
    print(f"   🔄 Running curation pipeline: {pipeline_name}")
    try:
        result = subprocess.run(
            ["python", str(pipeline_path)],
            cwd=str(curation_dir),
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Curation pipeline failed: {e}")
        print(e.stderr)
        return False

def validate_benchmark_manifest(manifest_path):
    """Validate that a benchmark manifest exists and has content"""
    if not os.path.exists(manifest_path):
        return False, "Manifest file not found"
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) == 0:
                return False, "Manifest is empty"
            
            # Validate first line is valid JSON
            first_entry = json.loads(lines[0])
            required_fields = ['audio_filepath', 'text', 'duration']
            missing = [f for f in required_fields if f not in first_entry]
            if missing:
                return False, f"Missing required fields: {missing}"
            
            return True, f"Valid manifest with {len(lines)} entries"
    except Exception as e:
        return False, f"Validation error: {e}"

def run_benchmark(model, manifest_path, output_dir, batch_size):
    """Run inference on a benchmark manifest"""
    print(f"   🚀 Running inference...")
    print(f"      Manifest: {manifest_path}")
    print(f"      Output: {output_dir}")
    
    # Read manifest
    audio_files = []
    ground_truths = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                audio_files.append(entry['audio_filepath'])
                ground_truths.append(entry['text'])
    
    print(f"      Files to transcribe: {len(audio_files)}")
    
    # Transcribe
    try:
        print(f"      Transcribing...")
        predictions = model.transcribe(audio_files, batch_size=batch_size)
        
        # Handle different return formats
        if isinstance(predictions, tuple):
            predictions = predictions[0]  # Some models return (texts, metadata)
        
        # Save predictions
        predictions_path = os.path.join(output_dir, 'predictions.json')
        results = []
        for i, (audio_file, pred, truth) in enumerate(zip(audio_files, predictions, ground_truths)):
            results.append({
                'audio_filepath': audio_file,
                'ground_truth': truth,
                'prediction': pred,
                'index': i
            })
        
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"      ✅ Transcription complete")
        return {
            'status': 'completed',
            'predictions_path': predictions_path,
            'num_samples': len(predictions)
        }
    
    except Exception as e:
        print(f"      ❌ Transcription failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }

def compute_metrics(predictions_path):
    """Compute WER/CER metrics"""
    print(f"   📊 Computing metrics...")
    
    if not JIWER_AVAILABLE:
        return {
            'wer': None,
            'cer': None,
            'status': 'error',
            'error': 'jiwer not installed'
        }
    
    try:
        # Load predictions
        with open(predictions_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        ground_truths = [r['ground_truth'] for r in results]
        predictions = [r['prediction'] for r in results]
        
        # Calculate WER and CER
        wer_score = wer(ground_truths, predictions) * 100  # Convert to percentage
        cer_score = cer(ground_truths, predictions) * 100  # Convert to percentage
        
        print(f"      WER: {wer_score:.2f}%")
        print(f"      CER: {cer_score:.2f}%")
        
        return {
            'wer': round(wer_score, 2),
            'cer': round(cer_score, 2),
            'num_samples': len(results),
            'status': 'completed'
        }
    
    except Exception as e:
        print(f"      ❌ Metrics computation failed: {e}")
        return {
            'wer': None,
            'cer': None,
            'status': 'failed',
            'error': str(e)
        }

def generate_report(benchmark_results, output_dir):
    """Generate aggregate benchmark report"""
    report_path = os.path.join(output_dir, 'benchmark_report.json')
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': benchmark_results,
        'summary': {
            'total_benchmarks': len(benchmark_results),
            'completed': sum(1 for b in benchmark_results if b.get('status') == 'completed'),
            'failed': sum(1 for b in benchmark_results if b.get('status') == 'failed')
        }
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Report saved to: {report_path}")
    return report

def main():
    args = parse_args()
    
    print("=" * 80)
    print("ASR BENCHMARK RUNNER")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Benchmark Set: {args.benchmark_set}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)
    
    # Setup paths
    benchmark_data_dir = Path(__file__).parent.parent / "data"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\n🔧 Loading ASR model...")
    try:
        model = nemo_asr.models.ASRModel.restore_from(args.model)
        model.eval()  # Set to evaluation mode
        print(f"   ✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return 1
    
    # Discover available benchmarks
    print("\n📋 Discovering benchmarks...")
    available_benchmarks = discover_benchmarks(str(benchmark_data_dir), args.benchmark_set)
    
    if not available_benchmarks:
        print(f"❌ No benchmarks found for version '{args.benchmark_set}'")
        print("\n💡 Available curation pipelines:")
        print("   - kn_clean_read.py")
        print("   - kn_en_codeswitch.py")
        # print("   - kn_conversational.py")  # TODO
        # print("   - en_clean_read.py")       # TODO
        print("\nRun curation pipelines first to generate benchmark data.")
        return 1
    
    # Filter benchmarks if specific ones requested
    if args.benchmarks:
        available_benchmarks = [
            b for b in available_benchmarks 
            if b['name'] in args.benchmarks
        ]
        if not available_benchmarks:
            print(f"❌ None of the requested benchmarks found: {args.benchmarks}")
            return 1
    
    print(f"Found {len(available_benchmarks)} benchmark(s):")
    for b in available_benchmarks:
        print(f"   ✅ {b['name']}")
    
    # Validate all manifests
    print("\n🔍 Validating benchmark manifests...")
    for benchmark in available_benchmarks:
        is_valid, message = validate_benchmark_manifest(benchmark['manifest'])
        status = "✅" if is_valid else "❌"
        print(f"   {status} {benchmark['name']}: {message}")
        benchmark['valid'] = is_valid
    
    # Run benchmarks
    print("\n🚀 Running benchmarks...")
    benchmark_results = []
    
    for benchmark in available_benchmarks:
        if not benchmark['valid']:
            print(f"\n⏭️  Skipping {benchmark['name']} (invalid manifest)")
            benchmark_results.append({
                'name': benchmark['name'],
                'status': 'skipped',
                'reason': 'Invalid manifest'
            })
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Benchmark: {benchmark['name']}")
        print(f"{'=' * 80}")
        
        benchmark_output_dir = os.path.join(args.output_dir, benchmark['name'])
        os.makedirs(benchmark_output_dir, exist_ok=True)
        
        # Run inference
        inference_result = run_benchmark(
            model,
            benchmark['manifest'],
            benchmark_output_dir,
            args.batch_size
        )
        
        # Compute metrics if inference succeeded
        if inference_result['status'] == 'completed':
            metrics = compute_metrics(inference_result['predictions_path'])
        else:
            metrics = {
                'wer': None,
                'cer': None,
                'status': 'skipped',
                'reason': 'Inference failed'
            }
        
        benchmark_results.append({
            'name': benchmark['name'],
            'manifest': benchmark['manifest'],
            'status': inference_result['status'],
            'metrics': metrics,
            'output_dir': benchmark_output_dir
        })
    
    # Generate report
    print(f"\n{'=' * 80}")
    print("GENERATING REPORT")
    print(f"{'=' * 80}")
    report = generate_report(benchmark_results, args.output_dir)
    
    print("\n✅ Benchmark run complete!")
    print(f"   Results saved to: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
