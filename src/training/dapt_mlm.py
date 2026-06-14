"""Domain-adaptive pretraining (DAPT) on DAIC-WOZ dialogue chunks via MLM.

Continues masked-language-model training of a BERT-family encoder on the
dialogue chunks of the OFFICIAL TRAIN SPLIT ONLY (107 subjects) — dev/test
text never touches the encoder, so downstream evaluation stays leakage-free
for every protocol. Output directory is a normal HF checkpoint usable as
--encoder_name in finetune_tcmil.py (or in the frozen pipeline).

Low-risk by construction: MLM on in-domain text adapts the encoder to
disfluent spoken dialogue without using any label.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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


def main():
    p = argparse.ArgumentParser(description="DAPT-MLM on DAIC-WOZ train chunks")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--encoder_name", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "dapt.log"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    log.info(f"Args: {vars(args)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
    )

    # TRAIN SPLIT ONLY — see module docstring.
    train_ivs = load_official_split(args.data_dir, "train", args.window, args.stride)
    texts = [c for iv in train_ivs for c in iv["chunks"]]
    log.info(f"{len(train_ivs)} train interviews -> {len(texts)} chunks")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = AutoModelForMaskedLM.from_pretrained(args.encoder_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)
    loader = DataLoader(ChunkTextDataset(texts, tokenizer, args.max_len),
                        batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / max(1, int(0.06 * total)),
                                 max(0.0, (total - s) / max(1, total))))

    amp_dtype = (torch.bfloat16 if device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else None)
    step = 0
    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in loader:
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
        log.info(f"epoch {epoch}: mean mlm loss {np.mean(losses):.4f}")

    # Save the *base* encoder weights (AutoModel-loadable).
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info(f"DAPT checkpoint saved to {out_dir}")


if __name__ == "__main__":
    main()
