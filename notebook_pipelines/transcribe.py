import os
import glob
import nemo.collections.asr as nemo_asr
import torch

# --- CONFIGURATION ---
BASE_DIR = "/mnt/data/nemo_experiment"
MODEL_PATH = os.path.join(BASE_DIR, "multi.nemo")
DATASET_DIR = os.path.join(BASE_DIR, "datasets/mini")

def find_audio_files(directory, limit=2):
    """Recursively find first N .wav files in the dataset directory"""
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                wav_files.append(os.path.join(root, file))
                if len(wav_files) >= limit:
                    return wav_files
    return wav_files

def main():
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 2. Load the Model
    print(f"Loading model from {MODEL_PATH}...")
    # Map to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=MODEL_PATH)
    asr_model.freeze() # We are only predicting, not training
    asr_model = asr_model.to(device) 
    print("Model loaded successfully.")

    # 3. Find Test Audio
    print("Looking for audio files...")
    audio_files = find_audio_files(DATASET_DIR)
    
    if not audio_files:
        print("No audio files found! Did the LibriSpeech download finish?")
        return

    print(f"Found {len(audio_files)} files to test.")

    # 4. Run Inference
    print("\n--- STARTING TRANSCRIPTION ---")
    # The transcribe method expects a list of paths
    # transcriptions = asr_model.transcribe(paths2audio_files=audio_files, batch_size=1)
    
    # Create a list of 'en' strings matching the number of audio files
    langs = ['en'] * len(audio_files)
    # Pass the 'lang' argument
    transcriptions = asr_model.transcribe(audio=audio_files, lang=langs, batch_size=1)

    # 5. Print Results
    for i, audio_path in enumerate(audio_files):
        print(f"\nFile: {audio_path}")
        # Note: transcriptions[0] is the list of text strings
        print(f"Transcription: {transcriptions[0][i]}")

    print("\n--- TEST COMPLETE ---")
    print("NOTE: If the transcription looks like garbage, that is EXPECTED.")
    print("We reset the decoder but haven't trained it yet.")

if __name__ == "__main__":
    main()

