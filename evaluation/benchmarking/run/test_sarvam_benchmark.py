#!/usr/bin/env python3
"""
ASR Benchmark Runner for Sarvam API

Runs ASR evaluation using Sarvam AI's speech-to-text API.
"""

import os
import sys
import argparse
import json
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Metrics imports
try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("⚠️  Warning: jiwer not installed. Install with: pip install jiwer")


# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ASR benchmark using Sarvam API"
    )
    parser.add_argument("--api-key", type=str, required=False,
                        help="Sarvam API key (or set SARVAM_API_KEY env var)")
    parser.add_argument("--manifest", type=str, 
                        default="/Users/chaitanyakartik/Projects/asr-finetuning/evaluation/benchmarking/data/v1/kn_clean_read.json",
                        help="Path to manifest file")
    parser.add_argument("--output-dir", type=str, 
                        default="models/results_sarvam_api",
                        help="Directory to save results")
    parser.add_argument("--model", type=str, default="saarika:v2.5",
                        help="Sarvam model to use")
    parser.add_argument("--language-code", type=str, default="kn-IN",
                        help="Language code (e.g., kn-IN)")
    return parser.parse_args()


# -------------------------
# Sarvam API
# -------------------------
def transcribe_with_sarvam(audio_path, api_key, model="saarika:v2.5", language_code="kn-IN"):
    """
    Transcribe audio using Sarvam API
    """
    url = "https://api.sarvam.ai/speech-to-text"
    
    headers = {
        "api-subscription-key": api_key
    }
    
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': (os.path.basename(audio_path), audio_file, 'audio/wav')
            }
            data = {
                'model': model,
                'language_code': language_code
            }
            
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            # Sarvam API returns transcript in 'transcript' field
            return result.get('transcript', '')
    
    except requests.exceptions.RequestException as e:
        print(f"   ❌ API Error for {audio_path}: {e}")
        return ""
    except Exception as e:
        print(f"   ❌ Error processing {audio_path}: {e}")
        return ""


# -------------------------
# Benchmark
# -------------------------
def run_benchmark(manifest_path, api_key, model, language_code, output_dir):
    print("🚀 Running inference with Sarvam API")
    print(f"   Manifest: {manifest_path}")
    print(f"   Model: {model}")
    print(f"   Language: {language_code}")
    print(f"   Output: {output_dir}")
    
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest not found: {manifest_path}")
        return None
    
    # Load manifest
    audio_files = []
    ground_truths = []
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Fix path: replace /mnt/data/asr-finetuning with actual project root
                audio_path = entry["audio_filepath"]
                if audio_path.startswith("/mnt/data/asr-finetuning"):
                    audio_path = audio_path.replace("/mnt/data/asr-finetuning", str(PROJECT_ROOT))
                audio_files.append(audio_path)
                ground_truths.append(entry["text"])
    
    print(f"   Files to transcribe: {len(audio_files)}")
    
    results = []
    
    for idx, (audio_path, truth) in enumerate(zip(audio_files, ground_truths)):
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"   ⚠️  Audio file not found: {audio_path}")
            results.append({
                "audio_filepath": audio_path,
                "ground_truth": truth,
                "prediction": "",
                "index": idx,
                "error": "File not found"
            })
            continue
        
        # Transcribe
        prediction = transcribe_with_sarvam(audio_path, api_key, model, language_code)
        
        results.append({
            "audio_filepath": audio_path,
            "ground_truth": truth,
            "prediction": prediction,
            "index": idx,
        })
        
        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(audio_files)}")
    
    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.json")
    
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ Inference complete")
    
    return predictions_path


# -------------------------
# Metrics
# -------------------------
def compute_metrics(predictions_path):
    print("📊 Computing metrics")
    
    if not JIWER_AVAILABLE:
        return {"wer": None, "cer": None}
    
    with open(predictions_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    refs = [r["ground_truth"] for r in results]
    hyps = [r["prediction"] for r in results]
    
    return {
        "wer": round(wer(refs, hyps) * 100, 2),
        "cer": round(cer(refs, hyps) * 100, 2),
        "num_samples": len(results),
    }


def generate_report(metrics, output_dir):
    """Generate benchmark report"""
    # Save JSON report
    report_path = os.path.join(output_dir, 'benchmark_report.json')
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'model': 'sarvam-api'
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save text report
    text_report_path = os.path.join(output_dir, 'report.txt')
    wer_val = metrics.get('wer', 0)
    cer_val = metrics.get('cer', 0)
    num_samples = metrics.get('num_samples', 0)
    
    report_text = f"WER: {wer_val:.2f}% | CER: {cer_val:.2f}% | Samples: {num_samples}"
    
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write(report_text + '\n')
    
    print(f"\n📄 JSON report saved to: {report_path}")
    print(f"📄 Text report saved to: {text_report_path}")
    return report


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv('SARVAM_API_KEY')
    if not api_key:
        print("❌ API key required. Provide via --api-key or SARVAM_API_KEY env var")
        return 1
    
    print("=" * 80)
    print("SARVAM API ASR BENCHMARK")
    print("=" * 80)
    print(f"Model:    {args.model}")
    print(f"Language: {args.language_code}")
    print(f"Manifest: {args.manifest}")
    print(f"Output:   {args.output_dir}")
    print("=" * 80)
    
    # Convert relative path to absolute
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    
    # Run inference
    predictions_path = run_benchmark(
        manifest_path=args.manifest,
        api_key=api_key,
        model=args.model,
        language_code=args.language_code,
        output_dir=output_dir,
    )
    
    if not predictions_path:
        return 1
    
    # Compute metrics
    metrics = compute_metrics(predictions_path)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"WER: {metrics['wer']}%")
    print(f"CER: {metrics['cer']}%")
    print(f"Samples: {metrics['num_samples']}")
    print("=" * 80)
    
    # Generate report
    generate_report(metrics, output_dir)
    
    print("\n✅ Benchmark complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
