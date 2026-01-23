import argparse
import json
import torch
import os
import jiwer
from tqdm import tqdm
import librosa
import numpy as np
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

# --- 1. IMPORT DECODER ---
try:
    from pyctcdecode import build_ctcdecoder
except ImportError:
    print("❌ Error: pyctcdecode not found. Run: pip install pyctcdecode")
    exit(1)

# --- 2. AGGREGATE TOKENIZER HANDLER (The Fix) ---
def get_raw_vocab(model):
    """
    Correctly extracts Raw BPE tokens from an AggregateTokenizer.
    """
    vocab_raw = []
    
    # 1. Use 'ids_to_tokens' to get the Raw BPE pieces (with underscores)
    # This is the specific API that works on AggregateTokenizers
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'ids_to_tokens'):
        vocab_size = model.tokenizer.vocab_size
        print(f"   🔍 Unpacking AggregateTokenizer (Size: {vocab_size})...")
        
        # We process in chunks to be faster, or one by one to be safe
        # AggregateTokenizer expects a list of IDs
        for i in range(vocab_size):
            try:
                # Returns a list like ["_hello"]
                tokens = model.tokenizer.ids_to_tokens([i])
                if tokens:
                    vocab_raw.append(tokens[0])
                else:
                    vocab_raw.append(str(i)) # Fallback
            except:
                vocab_raw.append(f"<unk_{i}>")
                
    elif hasattr(model.decoder, 'vocabulary'):
        vocab_raw = model.decoder.vocabulary
    else:
        raise AttributeError("Could not find vocabulary API.")

    # 2. Deduplicate (Rename duplicates so pyctcdecode doesn't crash)
    vocab_final = []
    seen_counts = {}
    
    for token in vocab_raw:
        if token in seen_counts:
            seen_counts[token] += 1
            vocab_final.append(f"{token}_dup{seen_counts[token]}")
        else:
            seen_counts[token] = 0
            vocab_final.append(token)
            
    return vocab_final

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--kenlm_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beam_width", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=1.5)
    return parser.parse_args()

def load_audio(path, target_sr=16000):
    try:
        audio, _ = librosa.load(path, sr=target_sr)
        return torch.tensor(audio, dtype=torch.float32), len(audio)
    except:
        return None, 0

def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Device: {device}")

    # 1. Load Model
    print(f"🔄 Loading Model: {args.model}")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(args.model)
    model.eval()
    model.freeze()
    model = model.to(device)

    # 2. Setup KenLM (With Aggregate Fix)
    print(f"🧠 Loading KenLM: {args.kenlm_model_path}")
    vocab = get_raw_vocab(model)
    print(f"   ✅ Raw Vocab Size: {len(vocab)} (Duplicates renamed)")
    
    decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=args.kenlm_model_path,
        alpha=args.alpha, 
        beta=args.beta,
    )

    # 3. Load Data
    filepaths, references = [], []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            filepaths.append(item['audio_filepath'])
            references.append(item.get('text', ''))

    # Subset 1/4
    limit = len(filepaths) // 4
    filepaths = filepaths[:limit]
    references = references[:limit]
    print(f"🎧 Processing {len(filepaths)} files (1/4 Subset)...")

    predictions = []
    num_batches = int(np.ceil(len(filepaths) / args.batch_size))
    
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, len(filepaths))
            batch_paths = filepaths[start_idx:end_idx]
            
            # Load Audio
            audio_tensors, audio_lengths = [], []
            for path in batch_paths:
                tensor, length = load_audio(path)
                if tensor is not None:
                    audio_tensors.append(tensor)
                    audio_lengths.append(length)
                else:
                    audio_tensors.append(torch.zeros(1600))
                    audio_lengths.append(1600)
            if not audio_tensors: continue

            max_len = max(audio_lengths)
            padded_audio = torch.zeros(len(audio_tensors), max_len, device=device)
            length_tensor = torch.tensor(audio_lengths, device=device, dtype=torch.long)
            for idx, wav in enumerate(audio_tensors):
                padded_audio[idx, :len(wav)] = wav.to(device)

            # NeMo Forward
            processed_signal, processed_len = model.preprocessor(
                input_signal=padded_audio, length=length_tensor
            )
            encoded, encoded_len = model.encoder(
                audio_signal=processed_signal, length=processed_len
            )
            log_probs = model.ctc_decoder(encoder_output=encoded)
            log_probs_cpu = log_probs.cpu().numpy()
            
            # Decode
            for j in range(log_probs_cpu.shape[0]):
                valid_time = int(encoded_len[j].item())
                logits = log_probs_cpu[j][:valid_time]
                text = decoder.decode(logits, beam_width=args.beam_width)
                predictions.append(text)

    # Metrics
    wer = jiwer.wer(references, predictions) * 100
    cer = jiwer.cer(references, predictions) * 100
    
    print("="*40)
    print(f"✅ RESULTS (1/4 Data Subset)")
    print(f"WER: {wer:.2f}% | CER: {cer:.2f}%")
    print("="*40)
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "report.txt"), "w") as f:
        f.write(f"WER: {wer:.2f} | CER: {cer:.2f}")

if __name__ == "__main__":
    run_eval(parse_args())
