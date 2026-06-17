"""DAPT-MLM v2 — domain-adaptive pretraining with a held-out MLM validation
split, best-by-val checkpoint selection, and early stopping.

Why v2 (see cluster/iteration_2/README.md): the iteration-1 run
(checkpoints/dapt_bge_large) trained 3 epochs and logged TRAIN loss only.
The loss was still falling steeply at the last epoch (6.16 -> 3.87 -> 3.19,
~-0.68/epoch) — i.e. the MLM was undertrained — and there was no validation
signal, so "train longer" could not be justified without risking overfit on a
tiny 107-interview corpus.

v2 fixes both, while keeping the SAME leakage guarantee as v1: the encoder
still only ever sees the OFFICIAL TRAIN SPLIT. The MLM validation set is an
INTERVIEW-LEVEL hold-out carved out of that train split (no dev/test text, and
no chunk from a val interview appears in MLM-train), so every downstream
protocol stays leakage-free.

Output is a normal HF checkpoint usable as --encoder_name in finetune_tcmil.py.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# src/training/dapt_mlm_v2.py -> repo root is three parents up.
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.tcmil_data import load_official_split


class ChunkTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.enc = tokenizer(texts, truncation=True, max_length=max_len,
                             padding="max_length", return_tensors="pt")

    def __len__(self):
        return self.enc["input_ids"].size(0)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, eval_seed):
    """Mean MLM loss on the val set. The collator masks randomly, so we reseed
    to a FIXED eval_seed before every pass -> identical masked positions each
    epoch -> val losses are directly comparable across epochs."""
    _seed_all(eval_seed)
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype else torch.enable_grad())
        with ctx:
            losses.append(model(**batch).loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    p = argparse.ArgumentParser(description="DAPT-MLM v2 (val + best-ckpt)")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--encoder_name", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)      # was 3; loss not plateaued
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--val_frac", type=float, default=0.1)  # interview-level MLM hold-out
    p.add_argument("--patience", type=int, default=3)      # early stop on val loss
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_seed", type=int, default=1234)  # fixed val masking
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "dapt.log"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    log.info(f"Args: {vars(args)}")

    _seed_all(args.seed)

    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
    )

    # TRAIN SPLIT ONLY. Hold out whole interviews for MLM val so no chunk from a
    # val interview leaks into MLM-train. dev/test text is never loaded.
    train_ivs = load_official_split(args.data_dir, "train", args.window, args.stride)
    rng = random.Random(args.seed)
    rng.shuffle(train_ivs)
    n_val = max(1, int(round(args.val_frac * len(train_ivs)))) if args.val_frac > 0 else 0
    val_ivs = train_ivs[:n_val]
    tr_ivs = train_ivs[n_val:]
    tr_texts = [c for iv in tr_ivs for c in iv["chunks"]]
    val_texts = [c for iv in val_ivs for c in iv["chunks"]]
    log.info(f"{len(tr_ivs)} train / {len(val_ivs)} val interviews -> "
             f"{len(tr_texts)} train / {len(val_texts)} val chunks")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = AutoModelForMaskedLM.from_pretrained(args.encoder_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)
    train_loader = DataLoader(ChunkTextDataset(tr_texts, tokenizer, args.max_len),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator)
    val_loader = (DataLoader(ChunkTextDataset(val_texts, tokenizer, args.max_len),
                             batch_size=args.batch_size, shuffle=False,
                             collate_fn=collator) if val_texts else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / max(1, int(0.06 * total)),
                                 max(0.0, (total - s) / max(1, total))))

    amp_dtype = (torch.bfloat16 if device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else None)

    best_val = float("inf")
    bad_epochs = 0
    step = 0
    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
                   if amp_dtype else torch.enable_grad())
            with ctx:
                loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                log.info(f"epoch {epoch} step {step}: mlm loss {np.mean(losses[-50:]):.4f}")
        tr_loss = float(np.mean(losses))

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, amp_dtype, args.eval_seed)
            log.info(f"epoch {epoch}: train {tr_loss:.4f} | val {val_loss:.4f}")
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                bad_epochs = 0
                model.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)
                log.info(f"  new best val {best_val:.4f} -> checkpoint saved")
            else:
                bad_epochs += 1
                log.info(f"  no improve ({bad_epochs}/{args.patience})")
                if bad_epochs >= args.patience:
                    log.info("early stop")
                    break
        else:
            log.info(f"epoch {epoch}: mean mlm loss {tr_loss:.4f}")

    # No val -> save final (matches v1 behavior). With val, best is already saved.
    if val_loader is None:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    log.info(f"DAPT v2 done. best val MLM loss {best_val:.4f}. ckpt -> {out_dir}")


if __name__ == "__main__":
    main()
