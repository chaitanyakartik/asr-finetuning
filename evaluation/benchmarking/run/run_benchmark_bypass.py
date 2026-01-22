#!/usr/bin/env python3
"""
ASR Benchmark Runner (Single Manifest, Manual RNNT)

Runs ASR evaluation on a single manifest using manual RNNT inference.

python evaluation/benchmarking/run/run_benchmark_bypass.py \
--model=training/models/kathbath_hybrid_h200_scaleup_phase2_final.nemo \
--manifest=evaluation/benchmarking/curation/evaluation/benchmarking/data/v1/kn_clean_read.json \
--output-dir=models/results_conf_100m_v2

"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# NeMo imports
import nemo.collections.asr as nemo_asr
import yaml

# Metrics imports
try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("⚠️  Warning: jiwer not installed. Install with: pip install jiwer")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))


# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ASR benchmark on a single manifest (manual RNNT)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .nemo model file")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to a single manifest (.json)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Unused (kept for compatibility)")
    return parser.parse_args()

# -------------------------
# Manifest validation
# -------------------------
def validate_benchmark_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        return False, "Manifest file not found"

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return False, "Manifest is empty"

            first = json.loads(lines[0])
            required = ["audio_filepath", "text", "duration"]
            missing = [k for k in required if k not in first]
            if missing:
                return False, f"Missing required fields: {missing}"

        return True, f"Valid manifest with {len(lines)} entries"
    except Exception as e:
        return False, f"Validation error: {e}"


# -------------------------
# Manual RNNT inference
# -------------------------
def run_benchmark(model, manifest_path, output_dir):
    import librosa
    import torch

    print("🚀 Running inference (manual RNNT path)")
    print(f"   Manifest: {manifest_path}")
    print(f"   Output:   {output_dir}")

    device = next(model.parameters()).device

    audio_files = []
    ground_truths = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                audio_files.append(entry["audio_filepath"])
                ground_truths.append(entry["text"])

    print(f"   Files to transcribe: {len(audio_files)}")

    results = []

    for idx, (audio_path, truth) in enumerate(zip(audio_files, ground_truths)):
        try:
            audio, _ = librosa.load(audio_path, sr=16000)

            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
            audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(device)

            processed, processed_len = model.preprocessor(
                input_signal=audio_tensor,
                length=audio_len,
            )

            encoded, encoded_len = model.encoder(
                audio_signal=processed,
                length=processed_len,
            )

            with torch.no_grad():
                hyps = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded,
                    encoded_lengths=encoded_len,
                    return_hypotheses=True,
                )

            pred_text = hyps[0].text if hyps else ""

            results.append({
                "audio_filepath": audio_path,
                "ground_truth": truth,
                "prediction": pred_text,
                "index": idx,
            })

            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(audio_files)}")

        except Exception as e:
            print(f"   ❌ Failed on {audio_path}: {e}")
            results.append({
                "audio_filepath": audio_path,
                "ground_truth": truth,
                "prediction": "",
                "index": idx,
                "error": str(e),
            })

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
        'metrics': metrics
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save text report in the format requested
    text_report_path = os.path.join(output_dir, 'report.txt')
    wer = metrics.get('wer', 0)
    cer = metrics.get('cer', 0)
    num_samples = metrics.get('num_samples', 0)
    
    report_text = f"WER: {wer:.2f}% | CER: {cer:.2f}% | Samples: {num_samples}"
    
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

    print("=" * 80)
    print("ASR BENCHMARK RUNNER (SINGLE MANIFEST)")
    print("=" * 80)
    print(f"Model:    {args.model}")
    print(f"Manifest: {args.manifest}")
    print(f"Output:   {args.output_dir}")
    print("=" * 80)


    # Validate manifest
    ok, msg = validate_benchmark_manifest(args.manifest)
    if not ok:
        print(f"❌ Manifest invalid: {msg}")
        return 1
    print(f"✅ Manifest OK: {msg}")

    # Load model
    print("\n🔧 Loading ASR model...")
    try:
        model = nemo_asr.models.ASRModel.restore_from(args.model)
        model.eval()
        model.freeze()
        print(f"✅ Model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run inference
    predictions_path = run_benchmark(
        model=model,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )

    # Metrics
    metrics = compute_metrics(predictions_path)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(metrics)

    # Save report to models directory
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    generate_report(metrics, models_dir)

    print("\n✅ Benchmark complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
