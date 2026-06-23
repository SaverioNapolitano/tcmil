"""End-to-end encoder fine-tuning for TC-MIL (cluster experiments).

Trains the sentence encoder *through* the MIL objective: chunks are encoded
with gradients, mean-pooled, L2-normalized, and fed to the usual
GRU + gated-attention MIL head. Designed for the DAIC-WOZ regime where the
dataset (107 training subjects, ~3K chunks) is tiny relative to the encoder
(110M-335M), so every method here is a different point on the
"how much pretrained knowledge do we risk" axis:

    frozen   - control; must reproduce the frozen-pipeline numbers.
    bitfit   - bias terms only (~0.1% params).
    lora     - low-rank adapters on attention (optionally + FFN) via peft.
    last_k   - unfreeze the top k transformer layers, layer-wise lr decay.
    full     - everything trainable at a very low lr (boundary probe).

Anti-forgetting measures: head warmup (encoder frozen for the first
--head_warmup_epochs while the MIL head settles), low encoder lr with
linear warmup, layer-wise lr decay (--llrd), early stopping on the
selection split, gradient clipping.

Protocols (--protocol):
    official  - train 107 / select+earlystop dev 33 / optional --eval_test.
    kfold     - StratifiedGroupKFold over train+dev; the encoder is
                fine-tuned from scratch INSIDE EVERY FOLD (no pool-level
                fine-tuning: that would leak fold-test subjects into the
                encoder). testprev threshold by default.
    mc        - same with StratifiedShuffleSplit.
    export_oof- kfold over train+dev that exports pooled OOF probabilities
                and thresholds for the official protocol (the FT analogue
                of oof_threshold_official.py).

Leakage guarantees: subject-disjoint asserts on every split; the encoder
never sees text of any subject it is evaluated on; thresholds come from
dev, OOF, or unlabeled test score distributions (testprev) only.
"""

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.models.tcmil import TCMIL
from src.core.tcmil_data import assert_no_leakage, load_official_split
from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class FTTCMIL(nn.Module):
    """Trainable encoder -> mean pool -> L2 norm -> TCMIL head."""

    def __init__(self, args):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(args.encoder_name)
        if args.grad_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self._apply_ft_method(args)

        dim = self.encoder.config.hidden_size
        self.head = TCMIL(
            embedding_dim=dim, proj_dim=args.proj_dim, attn_dim=args.attn_dim,
            dropout=args.dropout, temporal=args.temporal,
            gru_layers=args.gru_layers,
        )

    def _apply_ft_method(self, args):
        m = args.ft_method
        if m == "lora":
            from peft import LoraConfig, get_peft_model

            # Match attention projections across architectures: BERT/XLM-R use
            # query/key/value, causal-LM embedders (Qwen2, Mistral) use
            # q_proj/k_proj/v_proj. peft matches by name suffix, so passing both
            # sets is safe — only the existing ones bind.
            present = {n.split(".")[-1] for n, _ in self.encoder.named_modules()}
            attn = [m for m in ("query", "key", "value",
                                "q_proj", "k_proj", "v_proj") if m in present]
            ffn = [m for m in ("intermediate.dense", "output.dense",
                               "gate_proj", "up_proj", "down_proj") if m in present]
            targets = attn + (ffn if args.lora_targets == "attn_ffn" else [])
            if not targets:
                raise RuntimeError(
                    f"No known attention modules found for LoRA in {args.encoder_name}; "
                    "inspect named_modules() and extend the target list.")
            alpha = args.lora_alpha if args.lora_alpha > 0 else 2 * args.lora_r
            cfg = LoraConfig(
                r=args.lora_r, lora_alpha=alpha,
                lora_dropout=args.lora_dropout, bias="none", target_modules=targets,
            )
            self.encoder = get_peft_model(self.encoder, cfg)
            return

        for p in self.encoder.parameters():
            p.requires_grad = False
        if m == "frozen":
            return
        if m == "bitfit":
            for n, p in self.encoder.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif m == "last_k":
            layers = self._layers()
            for layer in layers[-args.unfreeze_last_k:]:
                for p in layer.parameters():
                    p.requires_grad = True
        elif m == "full":
            for p in self.encoder.parameters():
                p.requires_grad = True
        else:
            raise ValueError(m)

    def _layers(self):
        enc = self.encoder
        base = getattr(enc, "base_model", enc)
        for attr in ("encoder", "transformer"):
            if hasattr(base, attr):
                return getattr(base, attr).layer
        raise RuntimeError("cannot locate transformer layers for last_k/llrd")

    def encode(self, input_ids, attention_mask, micro_batch: int, train: bool):
        """Encode a bag of chunks in micro-batches. (N, L) -> (N, d)."""
        outs = []
        for i in range(0, input_ids.size(0), micro_batch):
            ids = input_ids[i:i + micro_batch]
            am = attention_mask[i:i + micro_batch]
            out = self.encoder(input_ids=ids, attention_mask=am).last_hidden_state
            mask = am.unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            outs.append(F.normalize(pooled, p=2, dim=1))
        return torch.cat(outs, dim=0)

    def forward(self, input_ids, attention_mask, micro_batch=16, train=True):
        emb = self.encode(input_ids, attention_mask, micro_batch, train)
        bags = emb.unsqueeze(0)                       # (1, N, d)
        mask = torch.ones(1, emb.size(0), device=emb.device)
        return self.head(bags, mask)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def tokenize_interviews(interviews, tokenizer, max_len):
    for iv in interviews:
        enc = tokenizer(
            iv["chunks"], padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        iv["input_ids"] = enc["input_ids"]
        iv["attention_mask"] = enc["attention_mask"]
    return interviews


# --------------------------------------------------------------------------
# Optimizer with LLRD + head warmup
# --------------------------------------------------------------------------

def build_optimizer(model: FTTCMIL, args):
    """AdamW with separate head / encoder groups (+ layer-wise lr decay)."""
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": args.head_lr, "is_head": True}]

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc_params:
        if args.ft_method in ("full", "last_k") and args.llrd < 1.0:
            layers = model._layers()
            n_layers = len(layers)
            id2layer = {id(p): li for li, layer in enumerate(layers)
                        for p in layer.parameters()}
            buckets: dict[int | None, list] = {}
            for p in enc_params:
                buckets.setdefault(id2layer.get(id(p)), []).append(p)
            for li, params in buckets.items():
                # Non-layer params (embeddings, final norms) get the deepest lr.
                depth = (n_layers - 1 - li) if li is not None else n_layers
                groups.append({"params": params,
                               "lr": args.encoder_lr * (args.llrd ** depth),
                               "is_head": False})
        else:
            groups.append({"params": enc_params, "lr": args.encoder_lr,
                           "is_head": False})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args, steps_per_epoch):
    """Linear warmup/decay; encoder groups additionally held at lr=0 during
    the head-warmup epochs (per-group lambdas)."""
    total = max(1, args.max_epochs * steps_per_epoch)
    warm = int(args.warmup_ratio * total)
    head_warm_steps = args.head_warmup_epochs * steps_per_epoch

    def base(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return max(0.05, (total - step) / max(1, total - warm))

    lambdas = []
    for g in optimizer.param_groups:
        if g.get("is_head"):
            lambdas.append(base)
        else:
            lambdas.append(lambda s, b=base: 0.0 if s < head_warm_steps else b(s))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


# --------------------------------------------------------------------------
# Train / eval one seed
# --------------------------------------------------------------------------

from contextlib import nullcontext


def amp_ctx(device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


@torch.no_grad()
def predict_bags(model, interviews, device, micro_batch, amp_dtype):
    model.eval()
    probs, labels = [], []
    for iv in interviews:
        with amp_ctx(device, amp_dtype):
            out = model(iv["input_ids"].to(device), iv["attention_mask"].to(device),
                        micro_batch=micro_batch * 2, train=False)
        probs.append(torch.sigmoid(out["logits"].float()).item())
        labels.append(iv["label"])
    return np.array(probs), np.array(labels)


def train_one_seed_ft(args, seed, train_ivs, sel_ivs, device, log):
    """Fine-tune one model; early-stop on AUC over sel_ivs. Returns model."""
    set_seed(seed)
    model = FTTCMIL(args).to(device)
    optimizer = build_optimizer(model, args)
    sched = build_scheduler(optimizer, args, max(1, len(train_ivs) // args.accum))

    n_pos = sum(iv["label"] for iv in train_ivs)
    auto_pw = (len(train_ivs) - n_pos) / max(1, n_pos)
    pw = getattr(args, "pos_weight", 1.0)
    pos_weight = torch.tensor([auto_pw if pw < 0 else pw], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    sym_bce = nn.BCEWithLogitsLoss(reduction="none")

    amp_dtype = (torch.bfloat16 if device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else
                 torch.float16 if device.type == "cuda" else None)

    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    best_auc, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        if args.ft_method == "frozen":
            model.encoder.eval()
        warmup_active = epoch <= args.head_warmup_epochs

        order = list(range(len(train_ivs)))
        random.shuffle(order)
        optimizer.zero_grad()
        for step, idx in enumerate(order, 1):
            iv = train_ivs[idx]
            with amp_ctx(device, amp_dtype):
                out = model(iv["input_ids"].to(device), iv["attention_mask"].to(device),
                            micro_batch=args.micro_batch, train=True)
                label = torch.tensor([float(iv["label"])], device=device)
                loss = bce(out["logits"].float(), label)
                if args.aux_weight > 0 and iv["has_symptoms"]:
                    sym = torch.tensor(iv["symptoms"], device=device)
                    loss = loss + args.aux_weight * sym_bce(
                        out["symptom_logits"].float().squeeze(0), (sym >= 1).float()).mean()
            (loss / args.accum).backward()
            if step % args.accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                sched.step()
                optimizer.zero_grad()

        sel_probs, sel_labels = predict_bags(model, sel_ivs, device,
                                             args.micro_batch, amp_dtype)
        auc = compute_metrics(sel_labels, (sel_probs >= 0.5).astype(int), sel_probs)["roc_auc"]
        log.info(f"    seed {seed} epoch {epoch}: sel AUC={auc:.4f}"
                 f"{' (head warmup)' if warmup_active else ''}")
        if auc > best_auc:
            best_auc, no_improve = auc, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()
                          if k in trainable_names}
        else:
            no_improve += 1
        if no_improve >= args.patience and epoch > args.head_warmup_epochs:
            break

    if best_state is None:
        # No epoch improved over the init AUC (rare); snapshot current trainable.
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()
                      if k in trainable_names}
    model.load_state_dict(best_state, strict=False)
    return model, best_auc, amp_dtype, best_state


def write_ckpt_config(ckpt_dir: Path, args):
    """Persist the construction args needed to rebuild FTTCMIL and reload a
    saved (trainable-only) state dict via load_state_dict(..., strict=False)."""
    json.dump(
        {k: getattr(args, k) for k in (
            "encoder_name", "ft_method", "lora_r", "lora_targets",
            "unfreeze_last_k", "llrd", "proj_dim", "attn_dim", "dropout",
            "temporal", "gru_layers", "window", "stride", "max_len",
            "aux_weight", "pos_weight")},
        open(ckpt_dir / "config.json", "w"), indent=2)


def free(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Per-unit resume cache
# --------------------------------------------------------------------------
# Each protocol is a loop over independent training units (one seed for
# official; one fold x seed for cv). A wall-time kill mid-loop loses every
# completed unit because results.json is written only at the end. To resume
# "as close as possible to where it stopped", each unit's predictions+metrics
# are cached the moment it finishes; on a resubmit the finished units load
# from cache (no retrain) and only the missing ones run.

# Construction + protocol hyperparams that define a unit. If any of these
# change in the same output_dir, the cache is stale and is ignored.
_CFG_KEYS = (
    "protocol", "encoder_name", "ft_method", "lora_r", "lora_alpha",
    "lora_dropout", "lora_targets", "unfreeze_last_k", "llrd", "encoder_lr",
    "head_lr", "head_warmup_epochs", "warmup_ratio", "weight_decay",
    "max_epochs", "patience", "accum", "micro_batch", "temporal",
    "gru_layers", "pos_weight", "proj_dim", "attn_dim", "dropout",
    "aux_weight", "window", "stride", "max_len", "n_seeds", "base_seed",
    "seed", "n_folds", "n_splits", "test_size", "val_size", "threshold_mode",
    "threshold_metric", "max_train_bags",
)


def _cfg_key(args):
    return {k: getattr(args, k) for k in _CFG_KEYS}


def _load_unit(cache_dir, name, cfg_key, log):
    """Return a cached unit's data, or None if absent/corrupt/stale."""
    f = cache_dir / f"{name}.json"
    if not f.exists():
        return None
    try:
        d = json.load(open(f))
    except (json.JSONDecodeError, OSError):
        log.warning(f"corrupt cache {f.name}; recomputing")
        return None
    if d.get("cfg_key") != cfg_key:
        log.warning(f"stale cache {f.name} (config changed); recomputing")
        return None
    return d["data"]


def _save_unit(cache_dir, name, cfg_key, data):
    """Write a unit cache atomically (rename) so a kill mid-write can't leave
    a half-written file that a later run would treat as valid."""
    tmp = cache_dir / f"{name}.json.tmp"
    with open(tmp, "w") as f:
        json.dump({"cfg_key": cfg_key, "data": data}, f)
    tmp.replace(cache_dir / f"{name}.json")


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------

def run_official(args, device, log, out_dir):
    train_ivs = load_official_split(args.data_dir, "train", args.window, args.stride)
    dev_ivs = load_official_split(args.data_dir, "dev", args.window, args.stride)
    test_ivs = load_official_split(args.data_dir, "test", args.window, args.stride)
    assert_no_leakage(train_ivs, dev_ivs, test_ivs)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.encoder_name)
    for ivs in (train_ivs, dev_ivs, test_ivs):
        tokenize_interviews(ivs, tok, args.max_len)
    if args.max_train_bags:
        train_ivs = train_ivs[: args.max_train_bags]

    train_prev = float(np.mean([iv["label"] for iv in train_ivs]))
    ckpt_dir = None
    if args.save_weights:
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        write_ckpt_config(ckpt_dir, args)
    cfg_key = _cfg_key(args)
    cache_dir = out_dir / "seed_cache"
    if args.resume:
        cache_dir.mkdir(exist_ok=True)
    dev_runs, test_runs, per_seed = [], [], []
    dev_labels = test_labels = None
    for i in range(args.n_seeds):
        seed = args.base_seed + i
        cached = (_load_unit(cache_dir, f"seed_{seed}", cfg_key, log)
                  if args.resume else None)
        if cached is not None:
            dev_runs.append(np.array(cached["dev_probs"]))
            dev_labels = np.array(cached["dev_labels"])
            per_seed.append(cached["metrics"])
            if args.eval_test:
                test_runs.append(np.array(cached["test_probs"]))
                test_labels = np.array(cached["test_labels"])
            log.info(f"seed {seed}: loaded from cache (AUC={cached['metrics']['roc_auc']:.4f})")
            continue
        t0 = time.time()
        model, best_auc, amp, best_state = train_one_seed_ft(args, seed, train_ivs, dev_ivs, device, log)
        if ckpt_dir is not None:
            # Trainable params only (LoRA adapters / bias / unfrozen layers + MIL
            # head); reload onto a fresh FTTCMIL(config) with strict=False.
            torch.save(best_state, ckpt_dir / f"seed_{seed}.pt")
        dp, dev_labels = predict_bags(model, dev_ivs, device, args.micro_batch, amp)
        dev_runs.append(dp)
        m = compute_metrics(dev_labels, (dp >= 0.5).astype(int), dp)
        per_seed.append(m)
        log.info(f"seed {seed}: dev AUC={m['roc_auc']:.4f} ({time.time()-t0:.0f}s)")
        unit = {"dev_probs": dp.tolist(),
                "dev_labels": np.asarray(dev_labels).tolist(), "metrics": m}
        if args.eval_test:
            tp, test_labels = predict_bags(model, test_ivs, device, args.micro_batch, amp)
            test_runs.append(tp)
            unit["test_probs"] = tp.tolist()
            unit["test_labels"] = np.asarray(test_labels).tolist()
        if args.resume:
            _save_unit(cache_dir, f"seed_{seed}", cfg_key, unit)
        free(model)

    dev_avg = np.mean(dev_runs, axis=0)
    strategies = {
        "f1": tune_threshold(dev_labels, dev_avg, "f1"),
        "prevalence": tune_threshold(dev_labels, dev_avg, "prevalence", prevalence=train_prev),
    }
    if args.threshold_file:
        ext = json.load(open(args.threshold_file))
        strategies.update({f"oof_{k}": v for k, v in ext["thresholds"].items()})

    dev_ens = compute_metrics(dev_labels, (dev_avg >= strategies["prevalence"]).astype(int), dev_avg)
    seed_aucs = [m["roc_auc"] for m in per_seed]
    log.info(f"DEV ens AUC={dev_ens['roc_auc']:.4f} F1={dev_ens['f1']:.4f} "
             f"per-seed {np.mean(seed_aucs):.4f}±{np.std(seed_aucs):.4f}")

    results = {
        "args": vars(args), "per_seed_dev": per_seed, "dev_ensemble": dev_ens,
        "thresholds": strategies, "dev_probs": dev_avg.tolist(),
        "dev_labels": np.asarray(dev_labels).tolist(),
    }
    if args.eval_test:
        test_avg = np.mean(test_runs, axis=0)
        results["test_probs"] = test_avg.tolist()
        results["test_labels"] = np.asarray(test_labels).tolist()
        # Persist the per-seed test predictions and metrics so the official-test
        # table can report a per-seed mean+/-std AUC (matching tab:external/
        # tab:levers), not only the ensemble. test_prob_runs is the canonical
        # primitive (any per-seed metric is recomputable at any threshold);
        # test_per_seed metrics use the prevalence threshold for convenience.
        results["test_prob_runs"] = [tp.tolist() for tp in test_runs]
        pst = strategies.get("oof_prevalence") or strategies["prevalence"]
        results["test_per_seed"] = [
            compute_metrics(test_labels, (tp >= pst).astype(int), tp)
            for tp in test_runs]
        seed_test_aucs = [m["roc_auc"] for m in results["test_per_seed"]]
        results["test_by_strategy"] = {}
        for name, t in strategies.items():
            tm = compute_metrics(test_labels, (test_avg >= t).astype(int), test_avg)
            results["test_by_strategy"][name] = {"threshold": float(t), **tm}
            log.info(f"TEST [{name}] (t={t:.2f}): F1={tm['f1']:.4f} "
                     f"P={tm['precision']:.4f} R={tm['recall']:.4f} AUC={tm['roc_auc']:.4f}")
        log.info(f"TEST per-seed AUC {np.mean(seed_test_aucs):.4f}"
                 f"+/-{np.std(seed_test_aucs):.4f}")
    return results


def run_cv(args, device, log, out_dir):
    """kfold / mc / export_oof: encoder fine-tuned per fold (leakage-safe)."""
    pool = (load_official_split(args.data_dir, "train", args.window, args.stride)
            + load_official_split(args.data_dir, "dev", args.window, args.stride))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.encoder_name)
    tokenize_interviews(pool, tok, args.max_len)
    pool_prev = float(np.mean([iv["label"] for iv in pool]))

    labels = [iv["label"] for iv in pool]
    groups = [iv["interview_id"] for iv in pool]
    if args.protocol in ("kfold", "export_oof"):
        splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True,
                                        random_state=args.seed)
        split_iter = splitter.split(np.zeros(len(pool)), labels, groups)
    else:
        sss = StratifiedShuffleSplit(n_splits=args.n_splits, test_size=args.test_size,
                                     random_state=args.seed)
        split_iter = sss.split(np.zeros(len(pool)), labels)

    ckpt_dir = None
    if args.save_weights:
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        write_ckpt_config(ckpt_dir, args)
    cfg_key = _cfg_key(args)
    cache_dir = out_dir / "seed_cache"
    if args.resume:
        cache_dir.mkdir(exist_ok=True)

    raw, group_preds = [], {}
    oof_prob, oof_label = {}, {}
    pool_np = np.array(pool)

    for fold_idx, (tr_idx, te_idx) in enumerate(split_iter, 1):
        fold_train = pool_np[tr_idx].tolist()
        fold_test = pool_np[te_idx].tolist()
        te_ids = {iv["interview_id"] for iv in fold_test}
        assert te_ids.isdisjoint({iv["interview_id"] for iv in fold_train}), "LEAKAGE"

        # Fold-deterministic inner val for early stopping.
        sids = sorted(iv["interview_id"] for iv in fold_train)
        sid_lab = {iv["interview_id"]: iv["label"] for iv in fold_train}
        sss_in = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size,
                                        random_state=args.seed + fold_idx)
        tr_i, va_i = next(sss_in.split(np.zeros(len(sids)), [sid_lab[s] for s in sids]))
        va_ids = {sids[i] for i in va_i}
        inner_train = [iv for iv in fold_train if iv["interview_id"] not in va_ids]
        inner_val = [iv for iv in fold_train if iv["interview_id"] in va_ids]

        if args.max_train_bags:
            inner_train = inner_train[: args.max_train_bags]
        log.info(f"--- fold/split {fold_idx}: train {len(inner_train)} "
                 f"val {len(inner_val)} test {len(fold_test)}")
        group_preds[fold_idx] = []
        for s in range(args.n_seeds):
            run_seed = args.seed + fold_idx * 100 + s
            cached = (_load_unit(cache_dir, f"fold{fold_idx}_seed{run_seed}", cfg_key, log)
                      if args.resume else None)
            if cached is not None:
                m = cached["metrics"]
                raw.append(m)
                group_preds[fold_idx].append({
                    "probability": np.array(m["probability"]),
                    "true_label": np.array(m["true_label"]),
                    "val_probability": np.array(m["val_probability"]),
                    "val_true_label": np.array(m["val_true_label"])})
                log.info(f"  run seed={run_seed}: loaded from cache "
                         f"(AUC={m['roc_auc']:.4f} F1={m['f1']:.4f})")
                continue
            model, _, amp, best_state = train_one_seed_ft(args, run_seed, inner_train,
                                                          inner_val, device, log)
            if ckpt_dir is not None:
                torch.save(best_state, ckpt_dir / f"fold{fold_idx}_seed{run_seed}.pt")
            te_probs, te_labels = predict_bags(model, fold_test, device,
                                               args.micro_batch, amp)
            va_probs, va_labels = predict_bags(model, inner_val, device,
                                               args.micro_batch, amp)
            free(model)

            t = (tune_threshold(None, te_probs, metric="prevalence", prevalence=pool_prev)
                 if args.threshold_mode == "testprev" else
                 tune_threshold(va_labels, va_probs, metric=args.threshold_metric,
                                prevalence=pool_prev))
            m = compute_metrics(te_labels, (te_probs >= t).astype(int), te_probs)
            m.update({"_fold_idx": fold_idx, "_seed_idx": s, "_run_seed": run_seed,
                      "probability": te_probs.tolist(), "true_label": te_labels.tolist(),
                      "val_probability": va_probs.tolist(),
                      "val_true_label": va_labels.tolist()})
            raw.append(m)
            group_preds[fold_idx].append({
                "probability": te_probs, "true_label": te_labels,
                "val_probability": va_probs, "val_true_label": va_labels})
            if args.resume:
                _save_unit(cache_dir, f"fold{fold_idx}_seed{run_seed}", cfg_key, {"metrics": m})
            log.info(f"  run seed={run_seed}: AUC={m['roc_auc']:.4f} F1={m['f1']:.4f}")

        if args.protocol == "export_oof":
            avg = np.mean([p["probability"] for p in group_preds[fold_idx]], axis=0)
            for iv, pr in zip(fold_test, avg):
                oof_prob[iv["interview_id"]] = float(pr)
                oof_label[iv["interview_id"]] = iv["label"]

    from src.core.utils.evaluation import _seed_ensemble_metrics
    from src.core.utils.stats import compute_aggregate_metrics, format_aggregate_report

    def ens_thr(gi, vy, vp, tp):
        if args.threshold_mode == "testprev":
            return tune_threshold(None, tp, metric="prevalence", prevalence=pool_prev)
        return tune_threshold(vy, vp, metric=args.threshold_metric, prevalence=pool_prev)

    ens_metrics = _seed_ensemble_metrics(group_preds, "_fold_idx", ens_thr)
    per_run_agg = compute_aggregate_metrics(raw)
    ens_agg = compute_aggregate_metrics(ens_metrics)
    report = (f"## Per-Run\n{format_aggregate_report(per_run_agg)}\n\n"
              f"## Seed-Ensemble\n{format_aggregate_report(ens_agg)}\n")
    log.info("\n" + report)
    (out_dir / "cv_report.txt").write_text(report)

    results = {"args": vars(args), "per_run_aggregate": per_run_agg,
               "ensemble_aggregate": ens_agg, "raw": raw}

    if args.protocol == "export_oof":
        ids = sorted(oof_prob)
        probs = np.array([oof_prob[i] for i in ids])
        labs = np.array([oof_label[i] for i in ids])
        oof_auc = compute_metrics(labs, (probs >= 0.5).astype(int), probs)["roc_auc"]
        thresholds = {
            "f1": tune_threshold(labs, probs, "f1"),
            "prevalence": tune_threshold(labs, probs, "prevalence", prevalence=pool_prev),
        }
        results["oof"] = {"auc": float(oof_auc), "thresholds": thresholds,
                          "probs": probs.tolist(), "labels": labs.tolist()}
        with open(out_dir / "oof_thresholds.json", "w") as f:
            json.dump({"oof_auc": float(oof_auc), "thresholds": thresholds}, f, indent=2)
        log.info(f"OOF export: AUC={oof_auc:.4f} thresholds={thresholds}")
    return results


def main():
    p = argparse.ArgumentParser(description="TC-MIL encoder fine-tuning (cluster)")
    p.add_argument("--protocol", default="official",
                   choices=["official", "kfold", "mc", "export_oof"])
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--encoder_name", default="BAAI/bge-large-en-v1.5",
                   help="HF id or local path (e.g. a DAPT checkpoint dir).")

    p.add_argument("--ft_method", default="lora",
                   choices=["frozen", "bitfit", "lora", "last_k", "full"])
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=-1,
                   help="LoRA alpha; -1 (default) uses 2*lora_r.")
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_targets", default="attn", choices=["attn", "attn_ffn"])
    p.add_argument("--unfreeze_last_k", type=int, default=2)
    p.add_argument("--llrd", type=float, default=0.8)
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--head_lr", type=float, default=3e-4)
    p.add_argument("--head_warmup_epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--accum", type=int, default=8, help="bags per optimizer step")
    p.add_argument("--micro_batch", type=int, default=16, help="chunks per encoder pass")
    p.add_argument("--grad_checkpointing", action="store_true", default=True)
    p.add_argument("--no_grad_checkpointing", dest="grad_checkpointing",
                   action="store_false")

    # Head (kept at the frozen-pipeline winners)
    p.add_argument("--temporal", default="gru", choices=["none", "gru", "transformer"])
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--pos_weight", type=float, default=1.0,
                   help="BCE pos_weight; 1.0 = no class weighting (the winning "
                        "single-model recipe), <0 = auto (neg/pos).")
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--attn_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--aux_weight", type=float, default=0.3)

    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256)

    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42, help="CV split seed")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--threshold_mode", default="testprev",
                   choices=["innerval", "testprev"])
    p.add_argument("--threshold_metric", default="f1")
    p.add_argument("--eval_test", action="store_true")
    p.add_argument("--threshold_file", default="",
                   help="oof_thresholds.json from --protocol export_oof")

    p.add_argument("--save_weights", action="store_true", default=True,
                   help="Save per-seed trainable weights (default on; "
                        "reproducibility / offline ensembling).")
    p.add_argument("--no_save_weights", dest="save_weights", action="store_false")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Cache each finished seed/fold-seed and skip it on a "
                        "resubmit (default on; lets a wall-time-killed run "
                        "continue near where it stopped).")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs, 12 train bags, 1 seed — install check only")
    p.add_argument("--max_train_bags", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        args.max_epochs, args.n_seeds, args.max_train_bags = 2, 1, 12
        args.head_warmup_epochs, args.patience = 0, 99

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    log.info(f"Args: {vars(args)}")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"device={device}")

    if args.protocol == "official":
        results = run_official(args, device, log, out_dir)
    else:
        results = run_cv(args, device, log, out_dir)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log.info(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
