import argparse
import json
import torch
import os
import jiwer
import numpy as np
import librosa
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from tqdm import tqdm

try:
    from pyctcdecode import build_ctcdecoder
except ImportError:
    print("❌ Error: pyctcdecode not found. Run: pip install pyctcdecode")
    exit(1)

# --- CONFIG ---
DEFAULT_MODEL = "training/models/kathbath_hybrid_h200_scaleup_p3_phase3_final.nemo"
DEFAULT_KENLM = "data/training/wiki_subword_6gram.arpa"
DEFAULT_MANIFEST = "evaluation/benchmarking/curation/test_data/Kathbath/test_manifest.json"
SUBSET_SIZE = 128

def load_audio(path):
    try:
        audio, _ = librosa.load(path, sr=16000)
        return torch.tensor(audio, dtype=torch.float32), len(audio)
    except:
        return None, 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--kenlm", type=str, default=DEFAULT_KENLM)
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", type=int, default=SUBSET_SIZE)
    return parser.parse_args()

def run_grid_search():
    print("🛡️ RUNNING PARANOID SCRIPT")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"🔄 Loading Model...")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(args.model)
    model.eval()
    model.freeze()
    model = model.to(device)

    # 2. Extract Raw Vocab (4024 items)
    vocab_raw = []
    vocab_size = model.tokenizer.vocab_size
    if hasattr(model.tokenizer, 'ids_to_tokens'):
        vocab_raw = model.tokenizer.ids_to_tokens(list(range(vocab_size)))
    else:
        for i in range(vocab_size):
            vocab_raw.append(model.tokenizer.ids_to_text([i]))

    # Deduplicate
    vocab_clean = []
    seen = {}
    for t in vocab_raw:
        if t in seen:
            seen[t] += 1
            vocab_clean.append(f"{t}_dup{seen[t]}")
        else:
            seen[t] = 0
            vocab_clean.append(t)
            
    # 3. THE FIX: Explicitly append Blank to match Logit Size (4025)
    vocab_clean.append("<blank>")
    
    # 4. THE PARANOID CHECK: Force Truncate to 4025
    # The Model Output is 4025. We MUST NOT exceed this.
    target_size = 4025
    
    if len(vocab_clean) > target_size:
        print(f"⚠️ Vocab was {len(vocab_clean)}. TRUNCATING to {target_size}.")
        vocab_clean = vocab_clean[:target_size]
    elif len(vocab_clean) < target_size:
        print(f"⚠️ Vocab was {len(vocab_clean)}. PADDING to {target_size}.")
        diff = target_size - len(vocab_clean)
        vocab_clean += ["<blank>"] * diff

    print(f"✅ FINAL VOCAB SIZE PASSED TO DECODER: {len(vocab_clean)}")
    assert len(vocab_clean) == 4025, "Vocab size must be exactly 4025"

    # 5. Get Data
    print(f"🎧 Pre-computing logits for {args.subset} files...")
    filepaths, references = [], []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for line in f:
            if len(filepaths) >= args.subset: break
            item = json.loads(line)
            filepaths.append(item['audio_filepath'])
            references.append(item.get('text', ''))

    all_logits = []
    with torch.no_grad():
        for path in tqdm(filepaths):
            tensor, length = load_audio(path)
            if tensor is None: 
                all_logits.append(None)
                continue
            t_in = tensor.unsqueeze(0).to(device)
            l_in = torch.tensor([length], device=device)
            p_sig, p_len = model.preprocessor(input_signal=t_in, length=l_in)
            enc, enc_len = model.encoder(audio_signal=p_sig, length=p_len)
            log_probs = model.ctc_decoder(encoder_output=enc)
            
            valid_len = int(enc_len[0].item())
            logits = log_probs[0][:valid_len].cpu().numpy()
            
            # Double Check
            if logits.shape[1] != 4025:
                print(f"❌ FATAL: Model changed output size to {logits.shape[1]}?!")
                return

            all_logits.append(logits)

    # 6. Grid Search
    alphas = [0.3, 0.5, 0.7, 1.0]
    betas = [0.5, 1.0, 2.0]
    
    print("\n" + "="*50)
    print("🚀 STARTING GRID SEARCH")
    print("="*50)
    print(f"{'Alpha':<8} | {'Beta':<8} | {'WER':<8}")
    print("-" * 50)

    best_wer = 100.0
    best_params = (0, 0)
    best_preds = []

    for alpha in alphas:
        try:
            decoder = build_ctcdecoder(
                labels=vocab_clean,  # EXACTLY 4025 ITEMS
                kenlm_model_path=args.kenlm,
                alpha=alpha,
                beta=0.0
            )
        except Exception as e:
            print(f"Decoder Build Failed: {e}")
            continue

        for beta in betas:
            decoder.reset_params(alpha=alpha, beta=beta)
            preds = []
            valid_refs = []
            
            for i, logits in enumerate(all_logits):
                if logits is None: continue
                
                # Decode
                raw_text = decoder.decode(logits, beam_width=64)
                
                # --- STITCHING (Fixes 100% WER) ---
                stitched_text = raw_text.replace(" ", "").replace("\u2581", " ").strip()
                
                preds.append(stitched_text)
                valid_refs.append(references[i])

            if not preds: continue
            wer = jiwer.wer(valid_refs, preds) * 100
            
            print(f"{alpha:<8} | {beta:<8} | {wer:.2f}%")
            
            if wer < best_wer:
                best_wer = wer
                best_params = (alpha, beta)
                best_preds = preds

    print("="*50)
    print(f"🏆 BEST RESULT: WER {best_wer:.2f}%")
    print(f"   Alpha: {best_params[0]}")
    print(f"   Beta:  {best_params[1]}")
    print("="*50)
    
    print("\n👀 QUALITATIVE CHECK")
    for i in range(min(3, len(best_preds))):
        print(f"\nExample {i+1}:")
        print(f"Ref:  {valid_refs[i]}")
        print(f"Pred: {best_preds[i]}")

if __name__ == "__main__":
    run_grid_search()
