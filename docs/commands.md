python evaluation/benchmarking/run/run_benchmark_kenlm.py \
  --model_path "training/models/kathbath_hybrid_h200_scaleup_p3_phase3_final.nemo" \
  --manifest "/mnt/data/asr-finetuning/evaluation/benchmarking/curation/test_data/Kathbath/test_manifest.json" \
  --kenlm_model_path "data/training/v3/kannada_4gram.arpa"

python evaluation/benchmarking/run/run_benchmark_bypass.py \
--model=/mnt/data/asr-finetuning/training/models/kathbath_hybrid_h200_scaleup_phase4_final.nemo \
--manifest=evaluation/benchmarking/curation/evaluation/benchmarking/data/v1/kn_clean_read.json \
--output-dir=models/results_conf_100m_v3

python training/train_adv.py \
  --phase 4 \
  --exp_name "kathbath_hybrid_h200_scaleup" \
  --train_manifest "data/training/v2.1/master_manifest.json" \
  --base_model "training/models/kathbath_hybrid_h200_scaleup_p3_phase3_final.nemo" \
  --epochs 30 \
  --batch_size 32 \
  --accumulate_grad 2


/mnt/data/asr-finetuning/training/experiments/kathbath_hybrid_h200_scaleup_phase4/2026-01-23_02-31-05/checkpoints/kathbath_hybrid_h200_scaleup_phase4.nemo

cp -v /mnt/data/asr-finetuning/training/models/kathbath_hybrid_h200_scaleup_phase4_final.nemo /mnt/data/kn_asr_nvidia/HybridRNNTCTC_100m_scaleup.nemo





{
  "timestamp": "2026-01-23T22:56:08.308475",
  "metrics": {
    "wer": 14.59,
    "cer": 2.86,
    "num_samples": 2062
  }
}


source nemo-env/bin/activate
python NeMo/tools/speech_data_explorer/data_explorer.py /Users/chaitanyakartik/Projects/asr-finetuning/predictions/conf_hybrid_trained_100m/predictions_ready.json