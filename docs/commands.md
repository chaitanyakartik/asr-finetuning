tmux new -s d1
source /mnt/data/asr-env/bin/activate


python training/train_adv.py \
  --phase 4 \
  --exp_name asr_hybrid_en_kn_balanced_run1 \
  --train_manifest /mnt/data/asr-finetuning/data/training/v3/mixed_kn_en_50_50.json \
  --base_model /mnt/data/asr-finetuning/training/models/kathbath_hybrid_h200_scaleup_phase4_final.nemo \
  --epochs 10 \
  --batch_size 32 \
  --lr 0.0002 \
  --accumulate_grad 1


python training/train_hi.py \
  --phase 0 \
  --exp_name asr_3lang_en_kn_hi_balanced \
  --train_manifest /mnt/data/asr-finetuning/data/training/v3/mixed_kn_en_hi_27.5_27.5_45_final.json \
  --base_model /mnt/data/asr-finetuning/training/models/asr_hybrid_en_kn_balanced_run1_phase4_final.nemo \
  --epochs 6 \
  --batch_size 32










rsync -avhP \
  --exclude='asr-env' \
  --exclude='*.lock' \
  -e "ssh -J ubuntu@47.29.24.119" \
  /mnt/data/asr-finetuning/ \
  neurodx@100.66.9.247:/dev/shm/chaitanya/asr-finetuning/





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

wc -l file.json




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






rm -rf evaluation/benchmarking/data/v2
unzip evaluation/benchmarking/data/v2.zip

python scripts/data_processing/regenerate_manifests.py valuation/benchmarking/data/v2 --dry-run

cat evaluation/benchmarking/data/v2/Kathbath/train_manifest.json > evaluation/benchmarking/data/v2/open_source_manifest.json
cat evaluation/benchmarking/data/v2/Indicvoices/train_manifest.json >> evaluation/benchmarking/data/v2/open_source_manifest.json
cat evaluation/benchmarking/data/v2/Vaani/train_manifest.json >> evaluation/benchmarking/data/v2/open_source_manifest.json

wc -l evaluation/benchmarking/data/v2/open_source_manifest.json



cp evaluation/benchmarking/data/v2/gok_call_recordings/train_manifest.json evaluation/benchmarking/data/v2/gok_call_recordings_manifest.json


python evaluation/benchmarking/run/run_benchmark_bypass.py \
--model=training/models/kathbath_hybrid_h200_scaleup_phase4_final.nemo \
--manifest=evaluation/benchmarking/data/v2/open_source_manifest.json \
--output-dir=models/test_data_v2_results/open_source_conf_100m_v3 \
--exp-name=open_source_conf_100m_v3

python evaluation/benchmarking/run/run_benchmark_bypass.py \
--model=training/models/kathbath_hybrid_h200_scaleup_phase4_final.nemo \
--manifest=evaluation/benchmarking/data/v2/gok_call_recordings_manifest.json \
--output-dir=models/test_data_v2_results/call_recordings_conf_100m_v3 \
--exp-name=call_recordings_conf_100m_v3

python /mnt/data/asr-finetuning/evaluation/benchmarking/run/run_benchmark_ai4b.py \
--model /mnt/data/asr-finetuning/models/indicconformer_stt_kn_hybrid_rnnt_large.nemo \
--manifest evaluation/benchmarking/data/v2/open_source_manifest.json \
--output-dir models/test_data_v2_results/open_source_ai4b_100m \
--decoder rnnt \
--lang-id kn

python /mnt/data/asr-finetuning/evaluation/benchmarking/run/run_benchmark_ai4b.py \
--model /mnt/data/asr-finetuning/models/indicconformer_stt_kn_hybrid_rnnt_large.nemo \
--manifest evaluation/benchmarking/data/v2/gok_call_recordings_manifest.json \
--output-dir models/test_data_v2_results/call_recordings_ai4b_100m \
--decoder rnnt \
--lang-id kn



# Using Indicvoices v2 dataset
.venv/bin/python evaluation/benchmarking/run/test_sarvam_benchmark.py \
  --manifest evaluation/benchmarking/data/v2/gok_call_recordings_manifest.json \
  --output-dir test_data_v2_results/call_recordings_sarvam

# Using gok_call_recordings
.venv/bin/python evaluation/benchmarking/run/test_sarvam_benchmark.py \
  --manifest evaluation/benchmarking/data/v2/open_source_manifest.json \
  --output-dir models/open_source_sarvam








python inference/asr_server.py