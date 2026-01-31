#!/usr/bin/env python3
"""
Process multiple audio files: split, transcribe, create manifests
"""
import os
import json
import base64
from pathlib import Path
from datetime import datetime
from pydub import AudioSegment
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

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
BASE_OUTPUT_DIR = "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/test_data"

# Audio files to process
AUDIO_FILES = [
    "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/Labour Department/Labour/Labour/214601.mp3",
    "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/Medical Education Department/Medical Education Department/Medical Education Department/230049-.mp3",
    "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/Minorites Welfare Department/Minorites Welfare Department/Minorites Welfare Department/216172-.mp3",
    "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/Public Works Department/PWD/PWD/216233.mp3",
    "/Users/chaitanyakartik/Projects/asr-finetuning/AI-ML/RDPR/RDPR/RDPR Data 1-3-25 to 1-05-25/228368RDPR.mp3"
]

# -------------------------
# Gemini Client Manager
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
# Step 1: Split Audio
# -------------------------
def split_audio(input_file, output_dir, chunk_duration_sec=30):
    """Split audio into chunks"""
    print(f"\n📂 Splitting: {os.path.basename(input_file)}")
    print(f"   Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    audio = AudioSegment.from_file(input_file)
    duration_ms = len(audio)
    duration_sec = duration_ms / 1000
    
    print(f"   Duration: {duration_sec:.2f}s")
    
    # Split into chunks
    chunk_length_ms = chunk_duration_sec * 1000
    chunk_files = []
    
    for i in range(0, duration_ms, chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_number = (i // chunk_length_ms) + 1
        
        # Get base filename without extension
        base_name = Path(input_file).stem
        output_file = os.path.join(output_dir, f"{base_name}_part{chunk_number:02d}.mp3")
        
        chunk.export(output_file, format="mp3")
        chunk_files.append(output_file)
    
    print(f"   ✅ Created {len(chunk_files)} chunks")
    return chunk_files

# -------------------------
# Step 2: Transcribe with Gemini
# -------------------------
def transcribe_with_gemini(audio_path, model_name="gemini-2.5-flash"):
    """Transcribe using Gemini API"""
    client = get_genai_client(SA_KEY_PATH)
    if not client:
        return {"filename": os.path.basename(audio_path), "error": "Client init failed"}

    try:
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        mime_type = "audio/mpeg" if audio_path.endswith('.mp3') else "audio/wav"
        prompt = "Transcribe this audio file accurately. Provide only the transcription text in Kannada."
        
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
            "filename": os.path.basename(audio_path),
            "transcription": response.text,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"filename": os.path.basename(audio_path), "error": str(e)}

def transcribe_all_chunks(chunk_files, output_dir, max_workers=4):
    """Transcribe all chunks concurrently"""
    print(f"\n🤖 Transcribing {len(chunk_files)} chunks (concurrent: {max_workers})")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(transcribe_with_gemini, chunk_file): chunk_file 
            for chunk_file in chunk_files
        }
        
        # Process completed tasks
        for idx, future in enumerate(as_completed(future_to_file), 1):
            result = future.result()
            results.append(result)
            
            filename = result['filename']
            if "transcription" in result:
                print(f"   [{idx}/{len(chunk_files)}] ✅ {filename}")
            else:
                print(f"   [{idx}/{len(chunk_files)}] ❌ {filename}: {result.get('error', 'Unknown')}")
    
    # Sort results by filename to maintain order
    results.sort(key=lambda x: x['filename'])
    
    # Save transcriptions
    transcriptions_file = os.path.join(output_dir, "transcriptions_gemini_2.5_flash.json")
    with open(transcriptions_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Saved: {transcriptions_file}")
    return transcriptions_file

# -------------------------
# Step 3: Create Manifest
# -------------------------
def create_manifest(transcriptions_file, output_dir):
    """Create manifest file from transcriptions"""
    print(f"\n📝 Creating manifest")
    
    with open(transcriptions_file, 'r', encoding='utf-8') as f:
        transcriptions = json.load(f)
    
    manifest_entries = []
    
    for item in transcriptions:
        if "transcription" not in item:
            print(f"   ⚠️  Skipping {item['filename']} - no transcription")
            continue
        
        audio_path = os.path.join(output_dir, item['filename'])
        
        # Get duration
        try:
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
        except:
            duration = 0.0
        
        # Create manifest entry
        entry = {
            "audio_filepath": audio_path,
            "text": item['transcription'],
            "duration": round(duration, 7),
            "lang": "kn",
            "source": "govt_call_recording"
        }
        
        manifest_entries.append(entry)
    
    # Write manifest file
    manifest_file = os.path.join(output_dir, "manifest.json")
    with open(manifest_file, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Manifest created: {manifest_file}")
    print(f"   Total entries: {len(manifest_entries)}")
    return manifest_file

# -------------------------
# Main Pipeline
# -------------------------
def process_audio_file(input_file):
    """Process single audio file: split -> transcribe -> manifest"""
    if not os.path.exists(input_file):
        print(f"\n❌ File not found: {input_file}")
        return None
    
    # Get base filename for output directory
    base_name = Path(input_file).stem
    output_dir = os.path.join(BASE_OUTPUT_DIR, base_name)
    
    print("\n" + "="*80)
    print(f"Processing: {base_name}")
    print("="*80)
    
    try:
        # Step 1: Split
        chunk_files = split_audio(input_file, output_dir)
        
        # Step 2: Transcribe (concurrent)
        transcriptions_file = transcribe_all_chunks(chunk_files, output_dir, max_workers=4)
        
        # Step 3: Create manifest
        manifest_file = create_manifest(transcriptions_file, output_dir)
        
        print(f"\n✅ Complete: {base_name}")
        return manifest_file
        
    except Exception as e:
        print(f"\n❌ Error processing {base_name}: {e}")
        return None

def main():
    print("🚀 Starting batch audio processing")
    print(f"Files to process: {len(AUDIO_FILES)}")
    
    results = []
    for audio_file in AUDIO_FILES:
        result = process_audio_file(audio_file)
        results.append({
            "input_file": audio_file,
            "manifest": result,
            "success": result is not None
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"Processed: {success_count}/{len(results)}")
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        name = Path(r['input_file']).stem
        print(f"{status} {name}")
    
    print("\n✅ All processing complete!")

if __name__ == "__main__":
    main()
