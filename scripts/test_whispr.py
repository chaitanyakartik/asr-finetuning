#!/usr/bin/env python3
"""
Test Whisper ASR for Kannada
Usage: python test_whispr.py --audio <audio_file> [--model <model_name>] [--device <cuda|cpu>]
"""

import argparse
import sys
from pathlib import Path

try:
    from transformers import pipeline
except ImportError:
    print("❌ transformers not installed. Install with: pip install transformers torch")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Test Whisper ASR')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, default='openai/whisper-medium', 
                        help='Whisper model name or path (default: openai/whisper-medium)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--lang', type=str, default='kn', help='Language code (default: kn for Kannada)')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {args.audio}")
        sys.exit(1)
    
    print(f"🎙️  Audio: {args.audio}")
    print(f"🤖 Model: {args.model}")
    print(f"💻 Device: {args.device}")
    print(f"🌐 Language: {args.lang}")
    print()
    
    # Load model
    try:
        print("Loading Whisper model...")
        whisper_asr = pipeline(
            "automatic-speech-recognition", 
            model=args.model, 
            device=0 if args.device == 'cuda' else -1,
        )
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Configure language
    try:
        # Special case to handle languages not supported by whisper model
        if args.lang == 'or':  # Odia
            whisper_asr.model.config.forced_decoder_ids = (
                whisper_asr.tokenizer.get_decoder_prompt_ids(
                    language=None, task="transcribe"
                )
            )
        else:
            whisper_asr.model.config.forced_decoder_ids = (
                whisper_asr.tokenizer.get_decoder_prompt_ids(
                    language=args.lang, task="transcribe"
                )
            )
    except Exception as e:
        print(f"⚠️  Warning: Could not set language decoder: {e}")
        print("    Continuing without language-specific decoder...")
    
    # Transcribe
    try:
        print(f"\n🎵 Transcribing {audio_path.name}...")
        result = whisper_asr(str(audio_path))
        
        print("\n" + "="*80)
        print("TRANSCRIPTION")
        print("="*80)
        print(result["text"])
        print("="*80)
        #Then print the ground truth if available
        print(audio)
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()