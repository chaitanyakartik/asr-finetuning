#!/usr/bin/env python3
"""
Transcribe all audio chunks using Gemini 2.5 Flash
"""
import os
import json
import base64
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Service Account and API Setup
try:
    from google.oauth2 import service_account
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("⚠️  Google Cloud libraries not installed.")
    exit(1)

# Paths
SA_KEY_PATH = "/Users/chaitanyakartik/Projects/gok-ipgrs/wa/keys/gok-ipgrs-voice-sa.json"
AUDIO_DIR = "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/test_data/228016"

# -------------------------
# Gemini Client Manager (EXACT COPY)
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
# Transcription (EXACT COPY)
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

# -------------------------
# Main
# -------------------------
def main():
    print(f"🚀 Transcribing all files in: {AUDIO_DIR}")
    
    # Get all audio files
    audio_files = sorted([
        os.path.join(AUDIO_DIR, f) 
        for f in os.listdir(AUDIO_DIR) 
        if f.endswith(('.mp3', '.wav'))
    ])
    
    if not audio_files:
        print(f"❌ No audio files found in {AUDIO_DIR}")
        return
    
    print(f"Found {len(audio_files)} audio files\n")
    
    results = []
    
    for idx, audio_file in enumerate(audio_files, 1):
        filename = os.path.basename(audio_file)
        print(f"[{idx}/{len(audio_files)}] Processing: {filename}")
        
        result = transcribe_with_gemini(audio_file, model_name="gemini-2.5-flash")
        result["filename"] = filename
        results.append(result)
        
        if "transcription" in result:
            print(f"   ✅ Success: {result['transcription'][:80]}...")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown')}")
        print()
    
    # Save results
    output_file = os.path.join(AUDIO_DIR, "transcriptions_gemini_2.5_flash.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ All transcriptions saved to: {output_file}")
    
    # Print summary
    success_count = sum(1 for r in results if "transcription" in r)
    print(f"\nSummary: {success_count}/{len(results)} successful")

if __name__ == "__main__":
    main()
