import os
import torch
import nemo.collections.asr as nemo_asr
from functools import partial

# --- CONFIGURATION ---
BASE_DIR = "/mnt/data/nemo_experiment"
# POINT TO THE NEW TRAINED MODEL
MODEL_PATH = os.path.join(BASE_DIR, "multi_trained.nemo") 
DATASET_DIR = os.path.join(BASE_DIR, "datasets/mini")

def find_audio_files(directory, limit=2):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                wav_files.append(os.path.join(root, file))
                if len(wav_files) >= limit:
                    return wav_files
    return wav_files

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 1. Load the Model
    print(f"Loading model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Restore the model
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=MODEL_PATH)
    asr_model.freeze()
    asr_model = asr_model.to(device)
    print("Model loaded successfully.")

    # --- THE HOTWIRE FIX ---
    # The standard transcribe() function doesn't support multilingual arguments.
    # We monkey-patch the tokenizer to default to 'en' (English) for this test.
    # This prevents the "Expected 'lang' to be set" crash.
# --- THE BETTER HOTWIRE FIX ---
    # We patch text_to_ids, which is the actual function that demands 'lang'
    original_text_to_ids = asr_model.tokenizer.text_to_ids

    def hotwired_text_to_ids(*args, **kwargs):
        # Force 'lang' to be 'en' if not provided
        if 'lang' not in kwargs:
            kwargs['lang'] = 'en'
        return original_text_to_ids(*args, **kwargs)

    # Apply the patch
    asr_model.tokenizer.text_to_ids = hotwired_text_to_ids
    print("Tokenizer hotwired: text_to_ids defaults to 'en'.")
    # -----------------------
    # -----------------------

    # 2. Find Audio
    audio_files = find_audio_files(DATASET_DIR)
    if not audio_files:
        print("No audio files found.")
        return

    print(f"Testing on {len(audio_files)} files...")

    # 3. Run Inference (Standard call now works thanks to the hotwire)
    transcriptions = asr_model.transcribe(paths2audio_files=audio_files, batch_size=1)

    # 4. Print Results
    for i, audio_path in enumerate(audio_files):
        print(f"\nFile: {audio_path}")
        print(f"Transcription: {transcriptions[0][i]}")

if __name__ == "__main__":
    main()