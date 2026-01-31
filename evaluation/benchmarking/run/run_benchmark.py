#!/usr/bin/env python3
"""
ASR Benchmark Runner - Self-Healing Version

1. Auto-extracts and standardizes tokenizer paths.
2. Auto-detects configuration mismatches (deprecated flags).
3. If loading fails due to an "unexpected argument", it auto-removes the bad key and retries.
"""

import os
import sys
import argparse
import json
import tempfile
import tarfile
import shutil
import zipfile
import re
import yaml
from pathlib import Path

# NeMo imports
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

# Metrics imports
try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("⚠️  Warning: jiwer not installed. Install with: pip install jiwer")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description="Run ASR benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Path to .nemo model file")
    parser.add_argument("--benchmark-set", type=str, default="v1", help="Benchmark version to run")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None, help="Specific benchmarks to run")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    return parser.parse_args()

def discover_benchmarks(benchmark_dir, benchmark_set):
    version_dir = os.path.join(benchmark_dir, benchmark_set)
    if not os.path.exists(version_dir):
        print(f"❌ Benchmark set '{benchmark_set}' not found at {version_dir}")
        return []
    benchmarks = []
    for f in os.listdir(version_dir):
        if f.endswith('.json'):
            benchmarks.append({'name': f.replace('.json', ''), 'manifest': os.path.join(version_dir, f)})
    return benchmarks

def validate_benchmark_manifest(manifest_path):
    if not os.path.exists(manifest_path): return False, "Manifest file not found"
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines: return False, "Manifest is empty"
            first = json.loads(lines[0])
            missing = [k for k in ['audio_filepath', 'text', 'duration'] if k not in first]
            if missing: return False, f"Missing fields: {missing}"
            return True, f"Valid ({len(lines)} entries)"
    except Exception as e: return False, f"Error: {e}"

def run_benchmark(model, manifest_path, output_dir, batch_size):
    print(f"   🚀 Inference: {os.path.basename(manifest_path)}")
    audio_files, ground_truths = [], []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                audio_files.append(e['audio_filepath'])
                ground_truths.append(e['text'])
    
    try:
        predictions = model.transcribe(audio_files, batch_size=batch_size)
        if isinstance(predictions, tuple): predictions = predictions[0]
        
        results = []
        for i, (audio, pred, truth) in enumerate(zip(audio_files, predictions, ground_truths)):
            results.append({'audio_filepath': audio, 'ground_truth': truth, 'prediction': pred, 'index': i})
        
        out_path = os.path.join(output_dir, 'predictions.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return {'status': 'completed', 'predictions_path': out_path}
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        return {'status': 'failed', 'error': str(e)}

def compute_metrics(predictions_path):
    if not JIWER_AVAILABLE: return {'wer': None, 'status': 'error', 'error': 'jiwer missing'}
    try:
        with open(predictions_path, 'r', encoding='utf-8') as f: results = json.load(f)
        refs = [r['ground_truth'] for r in results]
        hyps = [r['prediction'] for r in results]
        w, c = wer(refs, hyps) * 100, cer(refs, hyps) * 100
        print(f"      WER: {w:.2f}% | CER: {c:.2f}%")
        return {'wer': round(w, 2), 'cer': round(c, 2), 'status': 'completed'}
    except Exception as e: return {'status': 'failed', 'error': str(e)}

def find_file_recursive(root_dir, extension=None, filename=None):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if filename and file == filename: return os.path.join(root, file)
            if extension and file.endswith(extension): return os.path.join(root, file)
    return None

def prune_config_key(config, bad_key):
    """Recursively delete a key from a nested dict."""
    deleted = False
    if isinstance(config, dict):
        if bad_key in config:
            del config[bad_key]
            deleted = True
        for k, v in config.items():
            if prune_config_key(v, bad_key):
                deleted = True
    elif isinstance(config, list):
        for item in config:
            if prune_config_key(item, bad_key):
                deleted = True
    return deleted

def main():
    args = parse_args()
    benchmark_data_dir = Path(__file__).parent.parent / "data"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup Temp Dir
    extract_base = "/mnt/data/tmp/nemo_extract"
    os.makedirs(extract_base, exist_ok=True)
    extract_dir = tempfile.mkdtemp(dir=extract_base)
    
    print(f"\n🔧 Initializing Robust Model Load...")
    
    try:
        # 1. EXTRACT ARCHIVE
        print(f"   📦 Extracting to: {extract_dir}")
        if tarfile.is_tarfile(args.model):
            with tarfile.open(args.model, 'r:*') as tar: tar.extractall(path=extract_dir)
        elif zipfile.is_zipfile(args.model):
            with zipfile.ZipFile(args.model, 'r') as z: z.extractall(path=extract_dir)

        # 2. LOCATE & NORMALIZE ARTIFACTS
        found_model = find_file_recursive(extract_dir, extension='.model')
        target_model = os.path.join(extract_dir, 'tokenizer.model')
        if found_model and found_model != target_model:
            shutil.copy2(found_model, target_model)

        found_vocab = find_file_recursive(extract_dir, filename='vocab.txt') or \
                      find_file_recursive(extract_dir, extension='.vocab')
        target_vocab = os.path.join(extract_dir, 'vocab.txt')
        has_vocab = False
        if found_vocab:
            has_vocab = True
            if found_vocab != target_vocab:
                shutil.copy2(found_vocab, target_vocab)

        # 3. INITIAL CONFIG PREP
        config_path = find_file_recursive(extract_dir, filename='model_config.yaml') or \
                      find_file_recursive(extract_dir, filename='model_config.json')
        if not config_path: raise FileNotFoundError("model_config not found")

        print(f"   🛠️  Preparing Config: {config_path}")
        with open(config_path, 'r') as f:
            try: config = yaml.safe_load(f)
            except: f.seek(0); config = json.load(f)

        # Basic Sanitization (Paths)
        def basic_sanitize(obj):
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if k == 'tokenizer' and isinstance(v, dict):
                        v['dir'] = extract_dir
                        if v.get('type') in ['sentencepiece', 'google_sentencepiece', 'multilingual']:
                            v['type'] = 'bpe'
                        v['model_path'] = target_model
                        if has_vocab: v['vocab_path'] = target_vocab
                        elif 'vocab_path' in v: del v['vocab_path']
                    else: basic_sanitize(v)
            elif isinstance(obj, list):
                for item in obj: basic_sanitize(item)
        
        basic_sanitize(config)
        
        override_path = os.path.join(extract_dir, 'override_config.yaml')
        with open(override_path, 'w') as f: yaml.dump(config, f)

        # 4. SELF-HEALING LOAD LOOP
        model = None
        MAX_RETRIES = 10
        print("   🔄 Instantiating ASR Model (Self-Healing Mode)...")
        
        for attempt in range(MAX_RETRIES):
            try:
                model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=args.model,
                    override_config_path=override_path
                )
                model.eval()
                print(f"   ✅ Success on attempt {attempt+1}!")
                break
            except Exception as e:
                err_str = str(e)
                # Regex to catch "unexpected keyword argument 'xyz'"
                match = re.search(r"unexpected keyword argument '([^']+)'", err_str)
                
                if match:
                    bad_arg = match.group(1)
                    print(f"      ⚠️  Attempt {attempt+1} failed: Found deprecated argument '{bad_arg}'")
                    print(f"      ✂️  Pruning '{bad_arg}' from config and retrying...")
                    
                    # Prune from config object
                    prune_config_key(config, bad_arg)
                    
                    # Update the override file
                    with open(override_path, 'w') as f: yaml.dump(config, f)
                else:
                    print(f"      ❌ Fatal Error on attempt {attempt+1}: {e}")
                    raise e
        
        if model is None:
            raise RuntimeError("Exceeded max retries for model healing.")

        print(f"   ✅ Model loaded: {type(model).__name__}")
        shutil.rmtree(extract_dir)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        if os.path.exists(extract_dir): shutil.rmtree(extract_dir)
        return 1

    # --- BENCHMARKING ---
    benchmarks = discover_benchmarks(str(benchmark_data_dir), args.benchmark_set)
    if args.benchmarks: benchmarks = [b for b in benchmarks if b['name'] in args.benchmarks]
    
    if not benchmarks:
        print("❌ No benchmarks found.")
        return 1

    print(f"\n📋 Found {len(benchmarks)} benchmark(s)")
    results = []
    for b in benchmarks:
        valid, msg = validate_benchmark_manifest(b['manifest'])
        if not valid:
            print(f"Skipping {b['name']}: {msg}")
            continue
            
        out_dir = os.path.join(args.output_dir, b['name'])
        os.makedirs(out_dir, exist_ok=True)
        res = run_benchmark(model, b['manifest'], out_dir, args.batch_size)
        metrics = compute_metrics(res['predictions_path']) if res['status'] == 'completed' else {}
        results.append({'name': b['name'], 'status': res['status'], 'metrics': metrics})

    report_path = os.path.join(args.output_dir, 'report.json')
    with open(report_path, 'w') as f: json.dump({'results': results}, f, indent=2)
    print(f"\n✅ Done. Report: {report_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
