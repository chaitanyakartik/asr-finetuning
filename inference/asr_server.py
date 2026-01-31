import os
import torch
import librosa
import nemo.collections.asr as nemo_asr
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# --- Configuration ---
# Set the path to your .nemo model here
MODEL_PATH = "training/models/kathbath_hybrid_h200_scaleup_phase4_final.nemo" 
# Choose your GPU (0 or 1)
DEVICE_ID = 0 
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="NeMo ASR Microservice")

# Enable CORS so your React app can talk to this server directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local prototype, allow all. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
asr_model = None

@app.on_event("startup")
def load_model():
    """Loads the model into GPU memory once on startup."""
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
        raise RuntimeError("Could not load ASR model")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to transcribe an uploaded audio file.
    Expects a file upload (multipart/form-data).
    """
    if not asr_model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Save uploaded file to a temporary file
    # Librosa needs a file path (or file-like object), saving to disk is safest for format detection
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 2. Load audio using Librosa (matches your benchmark script)
        # Resample to 16000Hz as required by most NeMo models
        audio, _ = librosa.load(tmp_path, sr=16000)

        # 3. Manual Inference Pipeline
        # Convert to tensor and move to GPU
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(DEVICE)

        # Preprocessor: raw audio -> acoustic features
        processed, processed_len = asr_model.preprocessor(
            input_signal=audio_tensor,
            length=audio_len,
        )

        # Encoder: features -> encoded representations
        encoded, encoded_len = asr_model.encoder(
            audio_signal=processed,
            length=processed_len,
        )

        # Decoder: encoded -> text predictions (RNNT)
        with torch.no_grad():
            hyps = asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_len,
                return_hypotheses=True,
            )

        pred_text = hyps[0].text if hyps else ""
        
        return {"transcription": pred_text}

    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    # Run on port 8000, listen on all interfaces
    uvicorn.run(app, host="0.0.0.0", port=8001)