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

def run_eval():
    print("👻 RUNNING FINAL EVAL WITH GHOST SPACE 👻")
    
    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Loading Model on {device}...")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(DEFAULT_MODEL)
    model.eval()
    model.freeze()
    model = model.to(device)

    # 2. Get Data
    print(f"🎧 Loading Manifest...")
    filepaths, references = [], []
    with open(DEFAULT_MANIFEST, 'r', encoding='utf-8') as f:
        for line in f:
            if len(filepaths) >= SUBSET_SIZE: break
            item = json.loads(line)
            filepaths.append(item['audio_filepath'])
            references.append(item.get('text', ''))

    # 3. Compute Logits
    print(f"🎧 Computing logits for {len(filepaths)} files...")
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
            
            # --- THE GHOST FIX ---
            # Pad logits with one extra column of -infinity
            # Shape becomes [Time, 4025] -> [Time, 4026]
            padding = np.full((logits.shape[0], 1), -100.0, dtype=logits.dtype)
            logits_padded = np.hstack([logits, padding])
            
            all_logits.append(logits_padded)

    # 4. BUILD VOCAB (4025 + AutoSpace)
    print("📚 Building Vocabulary...")
    vocab_raw = []
    vocab_size = model.tokenizer.vocab_size # 4024
    
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

    # Add BLANK explicitly (Index 4024)
    vocab_clean.append("<blank>")
    
    print(f"✅ Vocab Size: {len(vocab_clean)} (Library will add Space at 4026)")

    # 5. RUN DECODE
    print("\n🚀 STARTING DECODING")
    
    try:
        decoder = build_ctcdecoder(
            labels=vocab_clean, # 4025 items -> Library makes it 4026
            kenlm_model_path=DEFAULT_KENLM,
            alpha=0.5,
            beta=1.0
        )
    except Exception as e:
        print(f"❌ Decoder Crash: {e}")
        return

    preds = []
    valid_refs = []
    
    for i, logits in enumerate(all_logits):
        if logits is None: continue
        
        # Decode
        raw_text = decoder.decode(logits, beam_width=64)
        
        # Stitch
        # Since we use Subword KenLM, we expect "_he llo" output
        stitched_text = raw_text.replace(" ", "").replace("\u2581", " ").strip()
        
        preds.append(stitched_text)
        valid_refs.append(references[i])

    wer = jiwer.wer(valid_refs, preds) * 100
    print(f"\n🏆 FINAL RESULT (Alpha=0.5, Beta=1.0): WER {wer:.2f}%")
    
    print("\n👀 EXAMPLE:")
    if preds:
        print(f"Ref:  {valid_refs[0]}")
        print(f"Pred: {preds[0]}")

if __name__ == "__main__":
    run_eval()
