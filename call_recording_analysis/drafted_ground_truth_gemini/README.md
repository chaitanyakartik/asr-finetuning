# Drafted Ground Truth for Call Recordings using Gemini

This directory contains ground truth transcriptions generated using Gemini 2.5 Flash for government call recordings.

## Directory Structure

```
drafted_ground_truth_gemini/
├── audio_chunks/          # Audio files split into 30-second chunks
│   ├── 214601/           # Labour Department (26 chunks)
│   ├── 216172-/          # Minorites Welfare Department (36 chunks)
│   ├── 216233/           # Public Works Department (19 chunks)
│   ├── 228016/           # Housing Department (16 chunks)
│   ├── 228368RDPR/       # RDPR (19 chunks)
│   └── 230049-/          # Medical Education Department (25 chunks)
│
├── transcriptions/        # Gemini transcriptions (JSON format)
│   ├── 214601_transcriptions.json
│   ├── 216172-_transcriptions.json
│   ├── 216233_transcriptions.json
│   ├── 228016_transcriptions.json
│   ├── 228368RDPR_transcriptions.json
│   └── 230049-_transcriptions.json
│
├── manifests/            # NeMo-format manifest files
│   ├── 214601_manifest.json
│   ├── 216172-_manifest.json
│   ├── 216233_manifest.json
│   ├── 228016_manifest.json
│   ├── 228368RDPR_manifest.json
│   ├── 230049-_manifest.json
│   └── manifest_master.json  # Combined manifest (141 entries)
│
└── scripts/              # Processing scripts
    ├── split_audio.py
    ├── transcribe_chunks.py
    ├── create_manifest.py
    └── process_batch_audio.py
```

## Statistics

- **Total Files Processed**: 6
- **Total Audio Chunks**: 141 (30-second segments)
- **Transcription Model**: Gemini 2.5 Flash
- **Language**: Kannada (kn)
- **Source**: Government department call recordings

## Processing Pipeline

1. **Split Audio** (`split_audio.py`)
   - Splits long audio files into 30-second chunks
   - Maintains audio quality (MP3 format)

2. **Transcribe** (`transcribe_chunks.py` / `process_batch_audio.py`)
   - Uses Gemini 2.5 Flash with service account authentication
   - Concurrent processing (4 workers)
   - Generates transcriptions in Kannada

3. **Create Manifest** (`create_manifest.py`)
   - Converts transcriptions to NeMo manifest format
   - Includes audio filepath, text, duration, language, and source

## Departments Covered

1. **Labour Department** (214601) - 26 chunks
2. **Minorites Welfare Department** (216172-) - 36 chunks
3. **Public Works Department** (216233) - 19 chunks
4. **Housing Department** (228016) - 16 chunks
5. **RDPR** (228368RDPR) - 19 chunks
6. **Medical Education Department** (230049-) - 25 chunks

## Usage

The master manifest file (`manifests/manifest_master.json`) can be used directly for:
- ASR model evaluation
- Fine-tuning datasets
- Benchmarking
- Data exploration with NeMo tools

## Notes

- All transcriptions are drafts and should be reviewed for accuracy
- Generated using Gemini 2.5 Flash (native audio capability)
- Service account: gok-ipgrs-voice-sa.json
- Project: certain-perigee-466307-q6
