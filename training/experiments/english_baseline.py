import os
import sys
import logging
import subprocess
import torch
import librosa
import argparse
import numpy as np

# NeMo Imports
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from omegaconf import DictConfig, OmegaConf

# --- CONFIGURATION ---
# We define paths relative to where this script is running
# Assuming script is in training/experiments/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "baseline_data") # Specific folder for this test
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- PATCH: FIX 'KeyError: dir' ---
# Essential patch for your specific NeMo environment
_original_setup = EncDecHybridRNNTCTCBPEModel._setup_monolingual_tokenizer
def patched_setup(self, tokenizer_cfg):
    if 'dir' not in tokenizer_cfg:
        if isinstance(tokenizer_cfg, DictConfig):
            OmegaConf.set_struct(tokenizer_cfg, False); tokenizer_cfg.dir = None; OmegaConf.set_struct(tokenizer_cfg, True)
        else: tokenizer_cfg['dir'] = None
    return _original_setup(self, tokenizer_cfg)
EncDecHybridRNNTCTCBPEModel._setup_monolingual_tokenizer = patched_setup


def run_command(command):
    """Helper to run shell commands from Python"""
    logger.info(f"Running command: {' '.join(command)}")
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

def find_audio_files(directory, limit=5):
    """Recursively find .wav or .flac files"""
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                audio_files.append(os.path.join(root, file))
                if len(audio_files) >= limit: return audio_files
    return audio_files

def step_download_english_data():
    """Downloads Mini LibriSpeech for English Baseline Testing"""
    logger.info(">>> STEP 1: Checking/Downloading English Data")

    # 1. Download LibriSpeech Script if missing
    libri_script = os.path.join(SCRIPTS_DIR, "get_librispeech_data.py")
    if not os.path.exists(libri_script):
        logger.info("Fetching get_librispeech_data.py...")
        run_command(["wget", "-O", libri_script, "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/dataset_processing/get_librispeech_data.py"])
    
    # 2. Check if data already exists to avoid re-downloading
    # Mini LibriSpeech extracts to a folder named 'mini'
    expected_data_path = os.path.join(DATA_DIR, "mini")
    if os.path.exists(expected_data_path) and len(find_audio_files(expected_data_path, 1)) > 0:
        logger.info(f"Data found at {expected_data_path}. Skipping download.")
        return expected_data_path

    # 3. Run LibriSpeech Download
    logger.info("Downloading Mini LibriSpeech...")
    # This script handles downloading and extracting
    run_command(["python", libri_script, "--data_root", DATA_DIR, "--data_sets", "mini"])
    
    return expected_data_path

def step_run_inference(data_root):
    """Runs the Manual Inference Loop on the downloaded data"""
    logger.info(">>> STEP 2: Running English Baseline Inference")
    
    # 1. Load Stock English Model
    MODEL_NAME = "stt_en_conformer_transducer_small"
    logger.info(f"Loading stock model: {MODEL_NAME}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=MODEL_NAME)
    model.freeze()
    model = model.to(device)
    logger.info("✅ Model loaded.")

    # 2. Find Files
    audio_files = find_audio_files(data_root, limit=5)
    if not audio_files:
        logger.error("No audio files found for inference!")
        return

    logger.info(f"🚀 Testing on {len(audio_files)} files...")
    print("\n" + "="*80)

    # 3. Manual Inference Loop (Bypassing broken transcribe())
    for audio_path in audio_files:
        try:
            # A. Load Audio
            audio_signal, sr = librosa.load(audio_path, sr=16000)
            
            # B. Prepare Tensors
            audio_tensor = torch.tensor(audio_signal, dtype=torch.float32).unsqueeze(0).to(device)
            audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(device)
            
            # C. Preprocess (Spectrogram)
            processed_signal, processed_len = model.preprocessor(
                input_signal=audio_tensor, 
                length=audio_len
            )
            
            # D. Encoder (Features)
            encoded, encoded_len = model.encoder(
                audio_signal=processed_signal, 
                length=processed_len 
            )
            
            # E. Decoder (Generate Text)
            with torch.no_grad():
                best_hyp = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded, 
                    encoded_lengths=encoded_len,
                    return_hypotheses=True 
                )
            
            print(f"File: {os.path.basename(audio_path)}")
            print(f"Prediction: {best_hyp[0].text}\n")

        except Exception as e:
            logger.error(f"Failed on {os.path.basename(audio_path)}: {e}")

    print("="*80 + "\n")

def main():
    # 1. Download Data
    data_path = step_download_english_data()
    
    # 2. Run Inference
    step_run_inference(data_path)

if __name__ == "__main__":
    main()

