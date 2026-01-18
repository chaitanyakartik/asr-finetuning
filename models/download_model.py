#!/usr/bin/env python3
"""
Download ASR models from HuggingFace Hub

Usage:
    python download_model.py --repo ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large
    python download_model.py --repo ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large --filename model.nemo
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files

def parse_args():
    parser = argparse.ArgumentParser(description="Download ASR models from HuggingFace")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Specific file to download (default: auto-detect .nemo file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: current directory)"
    )
    return parser.parse_args()

def find_nemo_file(repo_id, token):
    """Find .nemo file in the repository"""
    print(f"🔍 Scanning repository for .nemo files...")
    try:
        files = list_repo_files(repo_id, token=token)
        nemo_files = [f for f in files if f.endswith('.nemo')]
        
        if not nemo_files:
            print("❌ No .nemo files found in repository")
            return None
        
        if len(nemo_files) > 1:
            print(f"⚠️  Multiple .nemo files found:")
            for f in nemo_files:
                print(f"   - {f}")
            print(f"Using first file: {nemo_files[0]}")
        
        return nemo_files[0]
    
    except Exception as e:
        print(f"❌ Error scanning repository: {e}")
        return None

def download_model(repo_id, filename, output_dir, token):
    """Download model file from HuggingFace Hub"""
    print(f"📥 Downloading from {repo_id}...")
    print(f"   File: {filename}")
    print(f"   Output: {output_dir}")
    
    try:
        # Download file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=None,  # Use default cache
            local_dir=output_dir,
            local_dir_use_symlinks=False,  # Copy file instead of symlink
            token=token
        )
        
        print(f"✅ Download complete!")
        print(f"   Saved to: {downloaded_path}")
        return downloaded_path
    
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main():
    args = parse_args()
    
    print("=" * 80)
    print("HuggingFace Model Downloader")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("⚠️  Warning: HF_TOKEN not found in .env file")
        print("   Some repositories may require authentication")
        hf_token = None
    else:
        print("✅ HuggingFace token loaded")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(__file__).parent)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine filename
    if args.filename:
        filename = args.filename
    else:
        filename = find_nemo_file(args.repo, hf_token)
        if not filename:
            return 1
    
    # Download
    downloaded_path = download_model(args.repo, filename, output_dir, hf_token)
    
    if downloaded_path:
        # Get file size
        file_size = os.path.getsize(downloaded_path)
        size_mb = file_size / (1024 * 1024)
        size_gb = file_size / (1024 * 1024 * 1024)
        
        if size_gb >= 1:
            print(f"   Size: {size_gb:.2f} GB")
        else:
            print(f"   Size: {size_mb:.2f} MB")
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
