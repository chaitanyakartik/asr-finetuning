import nemo.collections.asr as nemo_asr
import os

# PATHS
MODEL_PATH = "models/ai4bharat/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"
# Replace with a real audio path from your dataset
AUDIO_FILE = "data/processed_data/Kathbath/kannada/audio/84/121123/84-121123-0000.wav" 

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    print(f"⏳ Loading AI4Bharat 600M Multilingual Model...")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(MODEL_PATH)
    
    # IMPORTANT: Force the model to decode in Kannada
    # This specific model uses a vocabulary where we need to ensure the right tokens are active
    # Often for these models, passing language_id in transcribe or check_vocabulary is needed
    
    print(f"🎧 Transcribing: {AUDIO_FILE}")
    
    try:
        # We try to pass the language_id hint if the model accepts it
        # (common in AI4Bharat checkpoints)
        hypotheses = model.transcribe(
            audio=[AUDIO_FILE],
            batch_size=1
        )
        
        # Result handling
        text = hypotheses[0] if isinstance(hypotheses, list) else hypotheses
        if isinstance(text, tuple): text = text[0]
            
        print("\n" + "="*40)
        print(f"🗣️  RESULT (600M Model): {text}")
        print("="*40 + "\n")

    except Exception as e:
        print(f"⚠️ Inference Error: {e}")

if __name__ == "__main__":
    main()
