# ASR Benchmarking System

## Overview

This benchmarking system provides **fixed, versioned, reproducible evaluation datasets** for Kannada and English ASR models. Benchmarks are curated once, frozen, and versioned to ensure comparable metrics across model iterations.

## Design Principles

1. **Immutability**: Once frozen, benchmark datasets never change
2. **Versioning**: All benchmarks are versioned (v1, v2, etc.)
3. **Separation**: Benchmark data must never appear in training sets
4. **Reproducibility**: Same model + same benchmark + same config = same metrics
5. **Diversity**: Each benchmark tests a different ASR challenge

## Benchmark Categories (v1)

### 1. kn_clean_read - Kannada Clean Read Speech

**Purpose**: Evaluate ASR performance on high-quality, scripted Kannada speech

**Characteristics**:
- **Language**: Kannada only
- **Speech Type**: Read speech from scripts
- **Audio Quality**: Clean, studio/controlled recording
- **Acoustic Conditions**: Low noise, clear articulation
- **Speaking Style**: Formal, careful pronunciation

**Source Dataset**: ai4bharat/kathbath (Kannada test split)

**Selection Criteria**:
- Test split only (never trained on)
- Clear audio with minimal background noise
- Complete, grammatically correct transcripts
- Representative of formal Kannada speech

**Use Case**: Baseline ASR accuracy on ideal conditions

**Key Metrics**:
- Word Error Rate (WER)
- Character Error Rate (CER) - more informative for Kannada

---

### 2. kn_en_codeswitch - Kannada-English Code-Switching

**Purpose**: Evaluate ASR on natural code-switched speech (Kannada + English mixing)

**Characteristics**:
- **Language**: Mixed Kannada-English (code-switching)
- **Speech Type**: Natural conversational speech
- **Audio Quality**: Field recordings, variable quality
- **Acoustic Conditions**: Real-world noise, dialectal variation
- **Speaking Style**: Natural, spontaneous, colloquial

**Source Dataset**: ARTPARK-IISc/Vaani-transcription-part (Kannada test split)

**Selection Criteria**:
- Test split only
- Contains both Kannada and English segments
- Represents real-world bilingual usage patterns
- Dialect diversity across Karnataka districts

**Transcript Cleaning**:
- Remove noise tags: `<laugh>`, `<noise>`, etc.
- Remove English translations in `{brackets}`
- Preserve actual spoken content (including English words)

**Use Case**: Real-world Kannada speech patterns, bilingual capability

**Key Metrics**:
- Mixed-language WER
- Per-language segmentation accuracy (if evaluated separately)
- CER for Kannada segments

---

### 3. kn_conversational - Kannada Conversational Speech

**Status**: 🚧 **Not yet implemented**

**Purpose**: Evaluate ASR on natural conversational Kannada speech

**Planned Characteristics**:
- Pure Kannada (minimal code-switching)
- Spontaneous, informal speech
- Dialectal variations
- Natural disfluencies, hesitations

**Potential Sources**: TBD

---

### 4. en_clean_read - English Clean Read Speech

**Status**: 🚧 **Not yet implemented**

**Purpose**: Evaluate ASR on clean English speech (bilingual model validation)

**Planned Characteristics**:
- English-only speech
- Clear, read speech
- Standard accents
- High audio quality

**Potential Sources**: TBD (LibriSpeech test-clean, Common Voice, etc.)

---

## Directory Structure

```
evaluation/benchmarking/
├── benchmarking_definitions.md      ← This file
├── curation/                        ← Scripts to generate benchmarks
│   ├── kn_clean_read.py            ✅ Implemented
│   ├── kn_en_codeswitch.py         ✅ Implemented
│   ├── kn_conversational.py        🚧 TODO
│   └── en_clean_read.py            🚧 TODO
├── data/                            ← Frozen benchmark manifests
│   └── v1/                          ← Version 1 benchmarks
│       ├── kn_clean_read.json      ← Kathbath test
│       ├── kn_en_codeswitch.json   ← Vaani test
│       ├── kn_conversational.json  🚧 TODO
│       └── en_clean_read.json      🚧 TODO
├── run/                             ← Benchmark execution scripts
│   └── run_benchmark.py            ← Main benchmark runner
├── scoring/                         🚧 TODO: Metrics computation
└── reports/                         🚧 TODO: Benchmark results
```

---

## Usage

### 1. Generate Benchmarks (Curation)

```bash
# Test with 10 samples
cd evaluation/benchmarking/curation
python kn_clean_read.py

# Full benchmark generation (production)
# Edit N_SAMPLES = -1 in the script, then:
python kn_clean_read.py
```

### 2. Run Benchmarks

```bash
cd evaluation/benchmarking/run
python run_benchmark.py \
  --model path/to/model.nemo \
  --benchmark-set v1 \
  --output-dir ../../reports/run_001
```

### 3. View Results

Results will be saved to the specified output directory with:
- Per-sample transcripts
- Aggregate WER/CER metrics
- Benchmark summary report

---

## Benchmark Manifest Format

All benchmarks use **NeMo JSON manifest format** (one JSON object per line):

```json
{"audio_filepath": "/absolute/path/to/audio.wav", "text": "ground truth transcript", "duration": 12.5, "lang": "kn", "source": "kathbath_clean"}
{"audio_filepath": "/absolute/path/to/audio2.wav", "text": "another transcript", "duration": 8.3, "lang": "kn", "source": "kathbath_clean"}
```

**Required Fields**:
- `audio_filepath`: Absolute path to audio file
- `text`: Ground truth transcript
- `duration`: Audio duration in seconds

**Optional Metadata**:
- `lang`: Language code (`kn`, `en`, `kn-en`)
- `source`: Source identifier for provenance

---

## Versioning Policy

- **v1**: Initial benchmark set (current)
- Future versions (v2, v3, ...) will be created when:
  - New datasets become available
  - Benchmark criteria are updated
  - Bug fixes require re-curation

**Important**: Never modify existing versioned benchmarks. Always create a new version.

---

## Metrics

### Primary Metrics

1. **Word Error Rate (WER)**: Standard for English and mixed-language evaluation
2. **Character Error Rate (CER)**: More informative for Kannada (agglutinative language)

### Secondary Metrics (Optional)

- Latency (RTF - Real-Time Factor)
- Throughput (audio hours processed per hour)
- Per-dialect breakdown (if metadata available)

---

## Data Provenance

All benchmark audio files and manifests track:
- Original dataset name
- Original split (test/dev)
- Curation date
- Curation script version
- Any preprocessing applied

This ensures full reproducibility and proper attribution.

---

## Future Work

- [ ] Implement `kn_conversational` benchmark
- [ ] Implement `en_clean_read` benchmark
- [ ] Add automatic statistical validation (duration distribution, transcript length, etc.)
- [ ] Create visualization tools for benchmark characteristics
- [ ] Add per-dialect evaluation for Vaani data
- [ ] Implement confidence interval computation for metrics
