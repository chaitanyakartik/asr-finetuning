import os
import torch
import librosa
import nemo.collections.asr as nemo_asr
import tempfile
import traceback  # <--- Added for detailed error logs
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration ---
MODEL_PATH = "training/models/kathbath_hybrid_h200_scaleup_phase2_final.nemo" 
DEVICE_ID = 0 
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="NeMo ASR Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

asr_model = None

@app.on_event("startup")
def load_model():
    global asr_model
    print(f"🔧 Loading ASR model from {MODEL_PATH} on {DEVICE}...")
    try:
        asr_model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
        asr_model.eval()
        asr_model.freeze()
        asr_model = asr_model.to(DEVICE)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not asr_model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Save uploaded file
    file_ext = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        file_size = len(content)

    print(f"📥 Received file: {file.filename} ({file_size} bytes)")

    try:
        if file_size == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        # 1. Load audio
        # Using native soundfile if available, fallback to audioread
        audio, sr = librosa.load(tmp_path, sr=16000)
        
        duration = librosa.get_duration(y=audio, sr=sr)
        print(f"   Audio loaded: {duration:.2f}s, Sample Rate: {sr}Hz")

        if duration < 0.1:
            raise ValueError("Audio is too short (< 0.1s)")

        # 2. Inference
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(DEVICE)

        processed, processed_len = asr_model.preprocessor(
            input_signal=audio_tensor,
            length=audio_len,
        )

        encoded, encoded_len = asr_model.encoder(
            audio_signal=processed,
            length=processed_len,
        )

        with torch.no_grad():
            hyps = asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_len,
                return_hypotheses=True,
            )

        pred_text = hyps[0].text if hyps else ""
        print(f"✅ Transcription: {pred_text}")
        
        return {"transcription": pred_text}

    except Exception as e:
        print("❌ Error during transcription:")
        traceback.print_exc()  # <--- This will print the full error stack to your console
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) # Ensure this port matches your React app
