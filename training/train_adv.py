import os
import argparse
import logging
import json
import torch
import torch.distributed as dist
import lightning.pytorch as ptl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils import exp_manager

# H200 Optimization
torch.set_float32_matmul_precision('medium')

# ==============================================================================
# MONKEY PATCH: ROBUST DECODING (Prevents Safe Token Collapse)
# ==============================================================================
import sentencepiece as spm
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
EN_TOK_DIR = os.path.join(PROJECT_ROOT, "training/tokenizers/en/tokenizer_spe_bpe_v128")
KN_TOK_DIR = os.path.join(PROJECT_ROOT, "training/tokenizers/kn_master_v3000")

# Fallback Tokenizer Logic
fallback_model_path = os.path.join(EN_TOK_DIR, "tokenizer.model")
if os.path.exists(fallback_model_path):
    fallback_sp = spm.SentencePieceProcessor(model_file=fallback_model_path)
    _original_tokens_to_text = AggregateTokenizer.tokens_to_text

    def patched_tokens_to_text(self, tokens, lang_id=None):
        try:
            return _original_tokens_to_text(self, tokens, lang_id)
        except Exception:
            try:
                if hasattr(tokens, 'tolist'): tokens = tokens.tolist()
                clean_tokens = [t for t in tokens if t > 2]
                return fallback_sp.decode(clean_tokens)
            except:
                return ""
    AggregateTokenizer.tokens_to_text = patched_tokens_to_text

# ==============================================================================
# PHASE 3 CALLBACK: DYNAMIC CTC SCALING
# ==============================================================================
class CTCRampupCallback(Callback):
    """
    Implements Phase 3 Schedule:
    Epoch 0-1: CTC = 0.05
    Epoch 2-3: CTC = 0.1
    Epoch 4+:  CTC = 0.3
    """
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        # Define Schedule
        if epoch < 2:
            new_ctc = 0.05
        elif epoch < 4:
            new_ctc = 0.1
        else:
            new_ctc = 0.3
            
        # Apply to Model
        if hasattr(pl_module, 'loss'):
            pl_module.loss.ctc_loss_weight = new_ctc
            pl_module.loss.warprnnt_nll_loss_weight = 1.0 - (new_ctc * 0.5)
            
            # Log on Rank 0
            if trainer.global_rank == 0:
                logging.info(f"Phase 3 Scheduler (Epoch {epoch}): CTC Weight set to {new_ctc}")

# ==============================================================================
# MAIN TRAINING LOGIC
# ==============================================================================
def ensure_manifest_tags(manifest_path, lang_id, rank):
    if rank == 0 and os.path.exists(manifest_path):
        needs_update = False
        with open(manifest_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            if line and ('lang' not in json.loads(line) or json.loads(line)['lang'] != lang_id):
                needs_update = True
        
        if needs_update:
            logging.info(f"Rank 0: Injecting 'lang': '{lang_id}'...")
            with open(manifest_path, 'r', encoding='utf-8') as f: lines = f.readlines()
            with open(manifest_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    d = json.loads(line)
                    d['lang'] = lang_id
                    f.write(json.dumps(d) + "\n")
    if dist.is_initialized(): dist.barrier()

def run_training(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # ---------------------------------------------------------
    # 1. SETUP MODEL BASED ON PHASE
    # ---------------------------------------------------------
    if args.phase == 1:
        # Phase 1: Fresh Start (Surgery)
        if local_rank == 0: logging.info("PHASE 1: Decoder Bootstrapping (Fresh Surgery)")
        model = EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="stt_en_fastconformer_hybrid_large_pc")
        
        # Surgery
        new_tokenizer_cfg = OmegaConf.create({'type': 'agg', 'langs': {}})
        new_tokenizer_cfg.langs['en'] = OmegaConf.create({'dir': EN_TOK_DIR, 'type': 'bpe'})
        new_tokenizer_cfg.langs['kn'] = OmegaConf.create({'dir': KN_TOK_DIR, 'type': 'bpe'})
        model.change_vocabulary(new_tokenizer_dir=new_tokenizer_cfg, new_tokenizer_type="agg")
        
        # FIX: Kill SpecAugment & Dropout for Phase 1 stability
        if hasattr(model, 'spec_augmentation'):
            model.spec_augmentation = None
            if local_rank == 0: logging.info("SAFETY: SpecAugment Removed")
        
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'dropout'):
            model.encoder.dropout = 0.0
            
        # FIX: Re-init Decoder Bias
        if hasattr(model.decoder, 'layers'):
            for layer in model.decoder.layers:
                for name, p in layer.named_parameters():
                    if 'bias' in name:
                        torch.nn.init.zeros_(p)
                    elif 'weight' in name:
                        torch.nn.init.xavier_uniform_(p)
            if local_rank == 0: logging.info("Decoder Weights Re-Initialized")
        
        # STRATEGY: Freeze Encoder Completely
        model.encoder.freeze()
        if local_rank == 0: logging.info("Encoder: FULLY FROZEN")

    elif args.phase in [2, 3]:
        # Phase 2 & 3: Load Previous Phase (Partial Unfreeze)
        if not args.base_model:
            raise ValueError(f"Phase {args.phase} requires --base_model (result from previous phase)")
            
        if local_rank == 0: logging.info(f"PHASE {args.phase}: Loading Checkpoint {args.base_model}")
        model = EncDecHybridRNNTCTCBPEModel.restore_from(args.base_model)
        
        # STRATEGY: Partial Unfreeze
        model.encoder.unfreeze()
        
        # ROBUST LAYER DETECTION (Fixes AttributeError)
        if hasattr(model.encoder, 'layers'):
            encoder_layers = model.encoder.layers
        elif hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'layers'):
            encoder_layers = model.encoder.encoder.layers
        else:
            raise AttributeError("Could not locate encoder layers (checked .layers and .encoder.layers)")

        total_layers = len(encoder_layers)
        layers_to_train = 2
        freeze_until = total_layers - layers_to_train
        
        # Freeze bottom layers
        for idx, layer in enumerate(encoder_layers):
            if idx < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True # Ensure top layers are trainable
        
        if local_rank == 0: 
            logging.info(f"Encoder: Unfrozen Top {layers_to_train} Blocks. Bottom {freeze_until} Frozen.")
            
        # Phase 3 Relax SpecAugment
        if args.phase == 3 and hasattr(model, 'spec_augmentation') and model.spec_augmentation is not None:
            if hasattr(model.spec_augmentation, 'freq_masks'): model.spec_augmentation.freq_masks = 1 
            if hasattr(model.spec_augmentation, 'time_masks'): model.spec_augmentation.time_masks = 2 
            if local_rank == 0: logging.info("PHASE 3: Relaxed SpecAugment (Lighter Masking).")

    elif args.phase == 4:
        # Phase 4: Full Finetuning (No Freezing)
        if not args.base_model:
            raise ValueError(f"Phase {args.phase} requires --base_model (result from previous phase)")
        
        if local_rank == 0: logging.info(f"PHASE 4: Full Finetuning (Loading {args.base_model})")
        model = EncDecHybridRNNTCTCBPEModel.restore_from(args.base_model)

        # STRATEGY: FULLY UNFREEZE
        model.encoder.unfreeze()
        model.decoder.unfreeze()
        if hasattr(model, 'joint'):
            model.joint.unfreeze()

        # Ensure all parameters have requires_grad=True
        for param in model.parameters():
            param.requires_grad = True
            
        if local_rank == 0: logging.info("PHASE 4: Model Fully Unfrozen.")

    # ---------------------------------------------------------
    # 2. CONFIGURE LOSS & STRATEGY
    # ---------------------------------------------------------
    if hasattr(model, 'loss'):
        if args.phase == 1:
            # Phase 1: Zero CTC
            model.loss.warprnnt_nll_loss_weight = 1.0
            model.loss.ctc_loss_weight = 0.0
            if local_rank == 0: logging.info("Loss: Pure RNNT (CTC=0.0)")
            
        elif args.phase == 2:
            # Phase 2: Low CTC (Alignment Pressure)
            model.loss.warprnnt_nll_loss_weight = 0.95
            model.loss.ctc_loss_weight = 0.05
            if local_rank == 0: logging.info("Loss: RNNT=0.95, CTC=0.05")
        
        elif args.phase == 4:
            # Phase 4: Standard Hybrid
            model.loss.warprnnt_nll_loss_weight = 0.8
            model.loss.ctc_loss_weight = 0.2
            if local_rank == 0: logging.info("Loss: RNNT=0.8, CTC=0.2 (Balanced Hybrid)")
            
        # Phase 3 handled by Callback

    model.change_decoding_strategy(decoder_type="rnnt")
    model.cur_decoder = "rnnt"
    if hasattr(model, 'spec_augmentation') and args.phase == 1:
        model.spec_augmentation = None # Disable for stability in Phase 1

    # ---------------------------------------------------------
    # 3. DATA LOADING (Safe Mode Filtering)
    # ---------------------------------------------------------
    train_ds = {
        'manifest_filepath': args.train_manifest,
        'sample_rate': 16000,
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 16,
        'pin_memory': True,
        'use_start_end_token': False,
        'min_duration': 3.0,      # PHASE 0 SAFETY (No short files)
        'max_duration': 20.0,
        'prefetch_factor': 4
    }
    model.setup_training_data(train_data_config=train_ds)

    # Validation
    val_files = [
        os.path.join(PROJECT_ROOT, "evaluation/benchmarking/data/v1/en_clean_read.json"),
        os.path.join(PROJECT_ROOT, "evaluation/benchmarking/data/v1/kn_clean_read.json")
    ]
    for v in val_files: ensure_manifest_tags(v, "en" if "en_" in v else "kn", local_rank)
    
    model.setup_multiple_validation_data(val_data_config={
        'manifest_filepath': val_files,
        'sample_rate': 16000, 'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 16
    })

    # ---------------------------------------------------------
    # 4. OPTIMIZER & SCHEDULER
    # ---------------------------------------------------------
    # LR Logic
    if args.phase == 1:
        lr = 0.0005
        optimizer_sched = {'name': 'CosineAnnealing', 'warmup_ratio': 0.1, 'min_lr': 1e-7}
    elif args.phase == 4:
        # Phase 4 Specific Configuration
        lr = 1e-3  # High Learning Rate
        optimizer_sched = {
            'name': 'CosineAnnealing', 
            'warmup_steps': 1000,   # Explicit steps
            'min_lr': 1e-9
        }
    else:
        # Phase 2/3
        lr = 0.0001
        optimizer_sched = {'name': 'CosineAnnealing', 'warmup_ratio': 0.1, 'min_lr': 1e-7}
    
    if args.lr: lr = args.lr # Override if provided
        
    optimizer_conf = {
        'name': 'adamw', 
        'lr': lr, 
        'weight_decay': 1e-3,
        'sched': optimizer_sched
    }
    
    if local_rank == 0: logging.info(f"Optimizer Config: {optimizer_conf}")
    model.setup_optimization(optimizer_conf)

    # ---------------------------------------------------------
    # 5. TRAINER
    # ---------------------------------------------------------
    callbacks = []
    if args.phase == 3:
        callbacks.append(CTCRampupCallback())

    exp_config = exp_manager.ExpManagerConfig(
        exp_dir=os.path.join(PROJECT_ROOT, "training/experiments"),
        name=f"{args.exp_name}_phase{args.phase}",
        checkpoint_callback_params=exp_manager.CallbackParams(
            monitor="val_wer", mode="min", save_top_k=2, save_last=True, always_save_nemo=True
        ),
        create_wandb_logger=True,
        wandb_logger_kwargs={"name": f"{args.exp_name}_p{args.phase}", "project": "kannada-asr-curriculum"}
    )
    
    trainer = ptl.Trainer(
        devices=2, 
        accelerator="gpu", 
        strategy="ddp",
        precision="bf16-mixed", 
        gradient_clip_val=0.5,
        max_epochs=args.epochs,
        logger=False, 
        enable_checkpointing=False,
        accumulate_grad_batches=args.accumulate_grad, # NEW ARGUMENT
        callbacks=callbacks
    )

    config = OmegaConf.structured(exp_config)
    exp_manager.exp_manager(trainer, config)
    model.set_trainer(trainer)
    trainer.fit(model)

    if local_rank == 0:
        save_path = os.path.join(PROJECT_ROOT, f"training/models/{args.exp_name}_phase{args.phase}_final.nemo")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_to(save_path)
        logging.info(f"PHASE {args.phase} COMPLETE. Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4], help="Curriculum Phase")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--base_model", type=str, help="Required for Phase 2/3/4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, help="Override default phase LR")
    parser.add_argument("--accumulate_grad", type=int, default=1, help="Gradient Accumulation Steps (Default: 1)")
    
    args = parser.parse_args()
    run_training(args)

