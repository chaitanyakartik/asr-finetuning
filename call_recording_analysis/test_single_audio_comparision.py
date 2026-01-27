#!/usr/bin/env python3
import os
import sys
import json
import requests
import tempfile
import base64
import logging
from pathlib import Path
from datetime import datetime
from pydub import AudioSegment
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Service Account and API Setup
try:
    from google.oauth2 import service_account
    from google.cloud import speech_v1
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("⚠️  Google Cloud libraries not installed.")

# Paths and Constants
SA_KEY_PATH = "/Users/chaitanyakartik/Projects/gok-ipgrs/wa/keys/gok-ipgrs-voice-sa.json"
AUDIO_FILE = "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/Housing Department/housing department/housing department/228016.mp3"

# -------------------------
# Gemini Client Manager (from your snippet)
# -------------------------
def get_genai_client(sa_path, project_id="certain-perigee-466307-q6"):
    try:
        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return genai.Client(
            vertexai=True,
            project=project_id,
            location="us-central1",
            credentials=creds
        )
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        return None

# -------------------------
# Helpers
# -------------------------
def convert_to_wav(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.export(temp_wav.name, format='wav')
        return temp_wav.name
    except Exception:
        return None

# -------------------------
# Transcription Services
# -------------------------

def transcribe_with_gemini(audio_path, model_name="gemini-2.5-flash-native-audio"):
    """Fixed Gemini implementation using the genai.Client pattern"""
    if not GOOGLE_AVAILABLE: return {"service": f"Gemini ({model_name})", "error": "Libraries missing"}
    
    client = get_genai_client(SA_KEY_PATH)
    if not client: return {"service": f"Gemini ({model_name})", "error": "Client init failed"}

    try:
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        mime_type = "audio/mpeg" if audio_path.endswith('.mp3') else "audio/wav"
        
        prompt = "Transcribe this audio file accurately. Provide only the transcription text in Kannada."
        
        # Fixed call using types.Part and types.Blob
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=audio_data, mime_type=mime_type)
                    ]
                )
            ]
        )
        
        return {
            "service": f"Gemini ({model_name})",
            "transcription": response.text,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"service": f"Gemini ({model_name})", "error": str(e)}

def task_sarvam(audio_path):
    """Sarvam Task with automatic chunking for files > 30s"""
    api_key = os.getenv('SARVAM_API_KEY')
    if not api_key: 
        return {"service": "Sarvam", "error": "Missing API Key"}
    
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": api_key}
    
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        
        # Sarvam REST API limit is 30 seconds
        if duration_ms <= 30000:
            with open(audio_path, 'rb') as f:
                files = {'file': (os.path.basename(audio_path), f, 'audio/mpeg')}
                data = {'model': 'saarika:v2.5', 'language_code': 'kn-IN'}
                res = requests.post(url, headers=headers, files=files, data=data)
                res.raise_for_status()
                return {"service": "Sarvam", "transcription": res.json().get('transcript', '')}
        else:
            print(f"   ⚠️  Sarvam: File is {duration_ms/1000:.1f}s. Splitting into chunks...")
            full_transcript = []
            # Split into 25s chunks to be safe
            for i in range(0, duration_ms, 25000):
                chunk = audio[i:i+25000]
                with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
                    chunk.export(tmp.name, format="mp3")
                    with open(tmp.name, 'rb') as f:
                        files = {'file': (f"chunk_{i}.mp3", f, 'audio/mpeg')}
                        data = {'model': 'saarika:v2.5', 'language_code': 'kn-IN'}
                        res = requests.post(url, headers=headers, files=files, data=data)
                        res.raise_for_status()
                        full_transcript.append(res.json().get('transcript', ''))
            
            return {"service": "Sarvam", "transcription": " ".join(full_transcript)}

    except Exception as e:
        return {"service": "Sarvam", "error": f"Status {getattr(e.response, 'status_code', 'Unknown')}: {str(e)}"}
    

def transcribe_with_google_stt(audio_path, language_code="kn-IN"):
    if not GOOGLE_AVAILABLE: return {"service": "Google STT", "error": "Libraries missing"}
    try:
        wav_path = convert_to_wav(audio_path)
        client = speech_v1.SpeechClient.from_service_account_json(SA_KEY_PATH)
        with open(wav_path, 'rb') as f:
            audio = speech_v1.RecognitionAudio(content=f.read())
        
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language_code,
            enable_automatic_punctuation=True,
        )
        response = client.recognize(config=config, audio=audio)
        text = " ".join([res.alternatives[0].transcript for res in response.results])
        return {"service": "Google STT", "transcription": text, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"service": "Google STT", "error": str(e)}

# -------------------------
# Main Execution (Concurrent)
# -------------------------
def main():
    print(f"🚀 Starting Concurrent Transcription for: {os.path.basename(AUDIO_FILE)}")
    
    # Define tasks
    tasks = [
        (transcribe_with_gemini, AUDIO_FILE, "gemini-2.5-flash"),
        (transcribe_with_gemini, AUDIO_FILE, "gemini-2.5-pro"),
        (task_sarvam, AUDIO_FILE),
        (transcribe_with_google_stt, AUDIO_FILE)
    ]

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Map tasks to executor
        futures = [executor.submit(t[0], *t[1:]) for t in tasks]
        for future in futures:
            results.append(future.result())

    # Save and Display
    output_file = "transcription_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*30 + " RESULTS " + "="*30)
    for res in results:
        status = "✅ Success" if "transcription" in res else "❌ Error"+res.get("error", "")
        print(f"[{res['service']}] {status}")
        if "transcription" in res:
            print(f"Text: {res['transcription'][:100]}...")
    
    print(f"\nFull results saved to {output_file}")

if __name__ == "__main__":
    main()