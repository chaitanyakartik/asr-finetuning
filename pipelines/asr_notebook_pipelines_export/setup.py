import argparse
import os
import sys
import logging
import json
import subprocess
import torch
import pytorch_lightning as ptl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from huggingface_hub import login

# --- USER CONFIGURATION ---
# PASTE YOUR HUGGING FACE TOKEN HERE (Or set 'HF_TOKEN' env variable)
HF_TOKEN = "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"

# --- PATH CONFIGURATION ---
BASE_DIR = "/mnt/data/nemo_experiment"
DATA_DIR = os.path.join(BASE_DIR, "datasets")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizers")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
MODEL_CHECKPOINT = "multi.nemo"

# Ensure base directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "experiment.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command(command):
    """Helper to run shell commands from Python"""
    logger.info(f"Running command: {' '.join(command)}")
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

# --- STEP 1: DATA DOWNLOAD ---
def step_download_data():
    logger.info(">>> STEP 1: Downloading & Preparing Data")
    
    # 1. Login to HuggingFace
    if HF_TOKEN == "PASTE_YOUR_TOKEN_HERE":
        logger.warning("WARNING: HF_TOKEN not set. CommonVoice download might fail.")
    else:
        login(token=HF_TOKEN)

    # 2. Download LibriSpeech Script
    libri_script = os.path.join(SCRIPTS_DIR, "get_librispeech_data.py")
    if not os.path.exists(libri_script):
        run_command(["wget", "-O", libri_script, "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/dataset_processing/get_librispeech_data.py"])
    
    # 3. Run LibriSpeech Download
    logger.info("Downloading Mini LibriSpeech...")
    run_command(["python", libri_script, "--data_root", os.path.join(DATA_DIR, "mini"), "--data_sets", "mini"])

    # 4. Download CommonVoice Script
    cv_script = os.path.join(SCRIPTS_DIR, "convert_hf_dataset_to_nemo.py")
    if not os.path.exists(cv_script):
        run_command(["wget", "-O", cv_script, "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/speech_recognition/convert_hf_dataset_to_nemo.py"])

    # 5. Run CommonVoice Download
    logger.info("Downloading CommonVoice Spanish (Test Split)...")
    run_command([
        "python", cv_script,
        "path=fsicoli/common_voice_21_0",
        f"output_dir={DATA_DIR}",
        "name=es",
        "split=test",
        "use_auth_token=True"
    ])
    
    logger.info("Data download complete.")

# --- STEP 2: MANIFEST PROCESSING ---
def step_process_manifests():
    logger.info(">>> STEP 2: Processing Manifests (Adding Language IDs)")
    
    def add_lang_to_manifest(in_path, out_path, lang):
        logger.info(f"Processing {in_path} -> {out_path} with lang={lang}")
        if not os.path.exists(in_path):
            logger.error(f"Input manifest not found: {in_path}")
            return
            
        with open(in_path, 'r') as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                data = json.loads(line)
                data['lang'] = lang
                f_out.write(json.dumps(data) + '\n')

    # Define paths
    libri_train = os.path.join(DATA_DIR, "mini/train_clean_5.json")
    libri_dev = os.path.join(DATA_DIR, "mini/dev_clean_2.json")
    
    # Locate CommonVoice manifest
    cv_base = os.path.join(DATA_DIR, "mozilla-foundation/common_voice_3_0/es/test")
    cv_manifest_name = "test_mozilla-foundation_common_voice_3_0_manifest.json" 
    cv_full_path = os.path.join(cv_base, cv_manifest_name)
    
    if not os.path.exists(cv_full_path):
         # Try to find it if name differs
         if os.path.exists(cv_base):
             possible_files = [f for f in os.listdir(cv_base) if f.endswith(".json")]
             if possible_files:
                 cv_full_path = os.path.join(cv_base, possible_files[0])

    # Create subsets for CV
    cv_train_json = os.path.join(DATA_DIR, "commonvoice_train_manifest.json")
    cv_dev_json = os.path.join(DATA_DIR, "commonvoice_dev_manifest_1000.json")

    logger.info("Splitting CommonVoice manifest...")
    os.system(f"head -1000 {cv_full_path} > {cv_dev_json}")
    os.system(f"tail -n +1001 {cv_full_path} > {cv_train_json}")

    # Add Language IDs
    add_lang_to_manifest(libri_train, os.path.join(DATA_DIR, "train_clean_5_en.json"), "en")
    add_lang_to_manifest(libri_dev, os.path.join(DATA_DIR, "dev_clean_2_en.json"), "en")
    add_lang_to_manifest(cv_train_json, os.path.join(DATA_DIR, "commonvoice_train_manifest_es.json"), "es")
    add_lang_to_manifest(cv_dev_json, os.path.join(DATA_DIR, "commonvoice_dev_manifest_1000_es.json"), "es")
    
    logger.info("Manifest processing complete.")

# --- STEP 3: TOKENIZER TRAINING ---
def step_train_tokenizer():
    logger.info(">>> STEP 3: Training Tokenizers")
    
    tok_script = os.path.join(SCRIPTS_DIR, "process_asr_text_tokenizer.py")
    if not os.path.exists(tok_script):
        run_command(["wget", "-O", tok_script, "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tokenizers/process_asr_text_tokenizer.py"])

    # Train Spanish Tokenizer
    run_command([
        "python", tok_script,
        f"--manifest={os.path.join(DATA_DIR, 'commonvoice_train_manifest_es.json')}",
        f"--data_root={os.path.join(TOKENIZER_DIR, 'es')}",
        "--vocab_size=128", "--tokenizer=spe", "--spe_type=bpe", "--spe_character_coverage=1.0"
    ])

    # Train English Tokenizer
    run_command([
        "python", tok_script,
        f"--manifest={os.path.join(DATA_DIR, 'train_clean_5_en.json')}",
        f"--data_root={os.path.join(TOKENIZER_DIR, 'en')}",
        "--vocab_size=128", "--tokenizer=spe", "--spe_type=bpe", "--spe_character_coverage=1.0"
    ])
    logger.info("Tokenizers trained.")

# --- STEP 4: MODEL SETUP ---
def step_setup_model():
    logger.info(">>> STEP 4: Setting up Multilingual Model")
    
    # Load base English model
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_small")
    
    # Configure new Aggregate Tokenizer
    new_tokenizer_cfg = OmegaConf.create({'type': 'agg', 'langs': {}})
    
    en_tok_dir = os.path.join(TOKENIZER_DIR, 'en', 'tokenizer_spe_bpe_v128')
    es_tok_dir = os.path.join(TOKENIZER_DIR, 'es', 'tokenizer_spe_bpe_v128')
    
    new_tokenizer_cfg.langs['en'] = OmegaConf.create({'dir': en_tok_dir, 'type': 'bpe'})
    new_tokenizer_cfg.langs['es'] = OmegaConf.create({'dir': es_tok_dir, 'type': 'bpe'})
    
    # Apply change
    logger.info("Changing model vocabulary...")
    asr_model.change_vocabulary(new_tokenizer_dir=new_tokenizer_cfg, new_tokenizer_type="agg")
    
    # Save
    save_path = os.path.join(BASE_DIR, MODEL_CHECKPOINT)
    asr_model.save_to(save_path)
    logger.info(f"Model saved to {save_path}")

# --- STEP 5: TRAINING LOOP ---
def step_train_model():
    logger.info(">>> STEP 5: Training Loop")
    
    model_path = os.path.join(BASE_DIR, MODEL_CHECKPOINT)
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Run --step setup_model first.")
        return

    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=model_path)
    
    # Freeze Encoder
    logger.info("Freezing encoder...")
    asr_model.encoder.freeze()
    
    # Define Manifest Paths
    train_manifests = [
        os.path.join(DATA_DIR, "train_clean_5_en.json"),
        os.path.join(DATA_DIR, "commonvoice_train_manifest_es.json")
    ]
    val_manifests = [
        os.path.join(DATA_DIR, "dev_clean_2_en.json"),
        os.path.join(DATA_DIR, "commonvoice_dev_manifest_1000_es.json")
    ]
    
    # Setup Trainer
    trainer = ptl.Trainer(
        devices=[0], 
        accelerator="gpu",
        max_epochs=5,
        accumulate_grad_batches=1,
        precision=16,
        log_every_n_steps=10,
        enable_checkpointing=False
    )
    
    asr_model.set_trainer(trainer)
    
    # Setup DataLoaders
    train_ds = {
        'manifest_filepath': train_manifests,
        'sample_rate': 16000,
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True
    }
    
    val_ds = {
        'manifest_filepath': val_manifests,
        'sample_rate': 16000,
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 4
    }
    
    asr_model.setup_training_data(train_data_config=train_ds)
    asr_model.setup_multiple_validation_data(val_data_config=val_ds)
    
    optimizer_conf = {
        'name': 'adamw', 'lr': 0.01, 'weight_decay': 0,
        'sched': {'name': 'CosineAnnealing', 'warmup_ratio': 0.10, 'min_lr': 1e-6}
    }
    asr_model.setup_optimization(optimizer_conf)
    
    logger.info("Starting training...")
    trainer.fit(asr_model)
    
    final_save_path = os.path.join(BASE_DIR, "multi_trained.nemo")
    asr_model.save_to(final_save_path)
    logger.info(f"Training complete. Saved to {final_save_path}")

# --- MAIN CONTROLLER ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeMo Multilingual ASR Pipeline")
    parser.add_argument(
        "--step", 
        type=str, 
        required=True,
        choices=["download", "manifests", "tokenizer", "setup_model", "train", "all"],
        help="Pipeline step to execute"
    )

    args = parser.parse_args()

    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    if args.step == "download" or args.step == "all":
        step_download_data()
    
    if args.step == "manifests" or args.step == "all":
        step_process_manifests()

    if args.step == "tokenizer" or args.step == "all":
        step_train_tokenizer()

    if args.step == "setup_model" or args.step == "all":
        step_setup_model()

    if args.step == "train" or args.step == "all":
        step_train_model()


