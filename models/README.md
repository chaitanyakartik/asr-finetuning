# Models Directory

This directory stores downloaded ASR model files.

## Downloading Models

Use the `download_model.py` script to download models from HuggingFace:

```bash
# Download Kannada hybrid model
python download_model.py --repo ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large

# Download with specific filename
python download_model.py \
  --repo ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large \
  --filename indicconformer_stt_kn_hybrid_rnnt_large.nemo

# Download to specific directory
python download_model.py \
  --repo ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large \
  --output-dir /path/to/models
```

## Authentication

For private or gated repositories, create a `.env` file in the project root:

```bash
HF_TOKEN=hf_your_token_here
```

## Available Models

### Kannada Models

- **indicconformer_stt_kn_hybrid_ctc_rnnt_large**
  - Repository: `ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large`
  - Size: ~500MB
  - Architecture: Conformer Hybrid CTC-RNNT
  - Languages: Kannada

### Multilingual Models

- **Indic Conformer 600M** (22 languages including Kannada)
  - Repository: TBD
  - Size: ~2.4GB
  - Languages: Multiple Indic languages

### English Models

- **Parakeet RNNT 1.1B**
  - Repository: `nvidia/parakeet-rnnt-1.1b`
  - Size: ~4GB
  - Languages: English

## File Structure

```
models/
├── README.md
├── download_model.py
├── indicconformer_stt_kn_hybrid_rnnt_large.nemo
└── .gitignore
```

## Git Tracking

Model files (`.nemo`) are excluded from git tracking due to their large size.
