"""TC-MIL data pipeline: topic-chunk instances for MIL depression detection.

Builds MIL bags where each instance is a *dialogue chunk* — a sliding window
of consecutive (interviewer turn, participant turn) exchanges rendered as
natural dialogue text. Single utterances ("yeah", "mhm") carry almost no
signal for a sentence encoder; multi-exchange chunks restore topical context
(sleep, mood, energy, ...) that PHQ-8 symptoms correlate with.

Official AVEC2017 split handling:
    - train: train_split_Depression_AVEC2017.csv (107 subjects)
    - dev:   dev_split_Depression_AVEC2017.csv   (35 subjects)
    - test:  full_test_split.csv                 (47 subjects)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.core.preprocess_raw import (
    merge_by_gap,
    merge_consecutive_turns,
    parse_raw_transcript,
)

_LABEL_FILES = {
    "train": ("train_split_Depression_AVEC2017.csv", "PHQ8_Binary"),
    "dev": ("dev_split_Depression_AVEC2017.csv", "PHQ8_Binary"),
    "test": ("full_test_split.csv", "PHQ_Binary"),
}

_SYMPTOM_COLS = [
    "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
    "PHQ8_Appetite", "PHQ8_Failure", "PHQ8_Concentrating", "PHQ8_Moving",
]

# E-DAIC (AVEC 2019) zero-shot test set. Split CSVs carry the binary label as
# PHQ_Binary; the per-item PHQ-8 subscores live in a separate detailed file
# (0-3 each), in the same symptom order as _SYMPTOM_COLS.
_EDAIC_LABEL_FILES = {
    "train": ("train_split.csv", "PHQ_Binary"),
    "dev": ("dev_split.csv", "PHQ_Binary"),
    "test": ("test_split.csv", "PHQ_Binary"),
}
_EDAIC_DETAILED = "Detailed_PHQ8_Labels.csv"
_EDAIC_SYMPTOM_COLS = [
    "PHQ_8NoInterest", "PHQ_8Depressed", "PHQ_8Sleep", "PHQ_8Tired",
    "PHQ_8Appetite", "PHQ_8Failure", "PHQ_8Concentrating", "PHQ_8Moving",
]


def build_chunks(
    transcript_path: str | Path,
    window: int = 4,
    stride: int = 2,
    participant_only: bool = False,
    gap_merge: float | None = None,
) -> list[str]:
    """Build dialogue-chunk instances from a raw transcript.

    A chunk covers `window` consecutive participant turns together with the
    interviewer turn that precedes each, rendered as:

        Interviewer: <text>
        Participant: <text>
        ...

    Overlapping windows (stride < window) give the MIL attention multiple
    views of each topic segment.

    `participant_only=True` drops the Interviewer lines — the bias-control
    setting of Burdisso et al. 2024 (interviewer prompts carry a dataset
    shortcut on DAIC-WOZ).

    `gap_merge` (seconds) switches turn segmentation from speaker-diarized
    merging to silence-gap merging of participant rows only: consecutive
    participant segments are joined into one turn unless separated by a pause
    >= `gap_merge`. This matches the un-diarized E-DAIC ASR pipeline
    (`preprocess_raw.merge_by_gap`), so a model trained on DAIC-WOZ with
    `gap_merge` set sees the same instance construction as the E-DAIC zero-shot
    test set. Implies participant-only (interviewer turns are not available
    under this segmentation).
    """
    if gap_merge is not None:
        turns = [t for t in parse_raw_transcript(transcript_path)
                 if t["speaker"] == "Participant"]
        turns = merge_by_gap(turns, gap_threshold=gap_merge)
        # No interviewer turns under gap segmentation -> participant-only.
        exchanges = [(None, t["value"]) for t in turns]
        participant_only = True
    else:
        turns = merge_consecutive_turns(parse_raw_transcript(transcript_path))

        # Pair each participant turn with the preceding interviewer turn (if any)
        exchanges = []
        pending_q = None
        for t in turns:
            if t["speaker"] == "Ellie":
                pending_q = t["value"]
            elif t["speaker"] == "Participant":
                exchanges.append((pending_q, t["value"]))
                pending_q = None

    if not exchanges:
        return []

    chunks = []
    for start in range(0, len(exchanges), stride):
        win = exchanges[start:start + window]
        if not win:
            break
        lines = []
        for q, a in win:
            if q and not participant_only:
                lines.append(f"Interviewer: {q}")
            lines.append(f"Participant: {a}")
        chunks.append("\n".join(lines))
        if start + window >= len(exchanges):
            break

    return chunks


def load_official_split(
    data_dir: str | Path,
    split: str,
    window: int = 4,
    stride: int = 2,
    participant_only: bool = False,
    gap_merge: float | None = None,
) -> list[dict]:
    """Load one official split with chunk instances and PHQ-8 symptom targets."""
    data_dir = Path(data_dir)
    fname, label_col = _LABEL_FILES[split]
    df = pd.read_csv(data_dir / "labels" / fname)

    interviews = []
    for _, row in df.iterrows():
        pid = int(row["Participant_ID"])
        path = data_dir / "raw" / f"{pid}_TRANSCRIPT.csv"
        if not path.exists():
            print(f"  [WARN] {split}: missing raw transcript for {pid}, skipping.")
            continue

        chunks = build_chunks(path, window=window, stride=stride,
                              participant_only=participant_only,
                              gap_merge=gap_merge)
        if not chunks:
            print(f"  [WARN] {split}: no chunks for {pid}, skipping.")
            continue

        has_symptoms = all(c in df.columns and not pd.isna(row[c]) for c in _SYMPTOM_COLS)
        symptoms = [float(row[c]) for c in _SYMPTOM_COLS] if has_symptoms else [0.0] * 8

        interviews.append({
            "interview_id": pid,
            "split": split,
            "label": int(row[label_col]),
            "chunks": chunks,
            "symptoms": symptoms,
            "has_symptoms": has_symptoms,
        })

    return interviews


def load_edaic_split(
    data_dir: str | Path,
    split: str,
    window: int = 4,
    stride: int = 2,
    gap_merge: float = 2.0,
) -> list[dict]:
    """Load one E-DAIC split as MIL bags for zero-shot transfer from DAIC-WOZ.

    E-DAIC transcripts are un-diarized participant-only ASR (no interviewer
    turns), so chunks are always built with silence-gap segmentation
    (`gap_merge`); pass the SAME `gap_merge`/`window`/`stride` used to train the
    DAIC-WOZ model so instance construction matches and the only shift measured
    is the domain shift. `data_dir` is the E-DAIC root (e.g. ``data/e-daic``).

    Per-item PHQ-8 subscores are attached when the detailed-label file is
    present (aligned to _SYMPTOM_COLS order); they are unused at inference but
    kept so the bag dict matches `load_official_split`'s schema.
    """
    data_dir = Path(data_dir)
    fname, label_col = _EDAIC_LABEL_FILES[split]
    df = pd.read_csv(data_dir / "labels" / fname)

    detailed_path = data_dir / "labels" / _EDAIC_DETAILED
    sym_by_pid: dict[int, list[float]] = {}
    if detailed_path.exists():
        ddf = pd.read_csv(detailed_path)
        if all(c in ddf.columns for c in _EDAIC_SYMPTOM_COLS):
            for _, r in ddf.iterrows():
                sym_by_pid[int(r["Participant_ID"])] = [
                    float(r[c]) for c in _EDAIC_SYMPTOM_COLS
                ]

    interviews = []
    for _, row in df.iterrows():
        pid = int(row["Participant_ID"])
        path = data_dir / "raw" / f"{pid}_TRANSCRIPT.csv"
        if not path.exists():
            print(f"  [WARN] edaic/{split}: missing raw transcript for {pid}, skipping.")
            continue

        chunks = build_chunks(path, window=window, stride=stride, gap_merge=gap_merge)
        if not chunks:
            print(f"  [WARN] edaic/{split}: no chunks for {pid}, skipping.")
            continue

        symptoms = sym_by_pid.get(pid)
        has_symptoms = symptoms is not None
        interviews.append({
            "interview_id": pid,
            "split": f"edaic_{split}",
            "label": int(row[label_col]),
            "chunks": chunks,
            "symptoms": symptoms if has_symptoms else [0.0] * 8,
            "has_symptoms": has_symptoms,
        })

    return interviews


@torch.no_grad()
def embed_chunks(
    interviews: list[dict],
    encoder_name: str,
    device: torch.device,
    cache_dir: str | Path = "cache/tcmil",
    max_len: int = 256,
    batch_size: int = 16,
    cache_tag: str = "",
    prefix: str = "",
    pooling: str = "mean",
) -> list[dict]:
    """Embed every chunk with a frozen sentence encoder; cache per interview.

    Token pooling (`mean` for BERT-style encoders, `last` for causal-LM
    embedders like Qwen3-Embedding) + L2 normalization. `prefix` is
    prepended to every chunk (e5-family encoders require "query: ");
    callers must fold prefix and pooling into cache_tag so variants never
    share a cache.
    """
    from transformers import AutoModel, AutoTokenizer

    cache_dir = Path(cache_dir) / (encoder_name.replace("/", "__") + cache_tag)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, encoder = None, None

    for iv in interviews:
        cache_path = cache_dir / f"{iv['interview_id']}.pt"
        if cache_path.exists():
            iv["embeddings"] = torch.load(cache_path, weights_only=True)
            continue

        if encoder is None:
            # Native load (no remote code): transformers 5 supports Qwen2/BERT/
            # XLM-R natively; gte-Qwen2's remote tokenizer is incompatible with
            # transformers 5, but the base Qwen2Model loads fine here.
            tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            encoder = AutoModel.from_pretrained(encoder_name).to(device).eval()
            if pooling == "last" and tokenizer.padding_side != "left":
                tokenizer.padding_side = "left"  # last real token = position -1

        embs = []
        for i in range(0, len(iv["chunks"]), batch_size):
            batch = [prefix + c for c in iv["chunks"][i:i + batch_size]]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_len, return_tensors="pt",
            ).to(device)
            out = encoder(**enc).last_hidden_state
            if pooling == "last":
                pooled = out[:, -1]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embs.append(F.normalize(pooled, p=2, dim=1).cpu())

        iv["embeddings"] = torch.cat(embs, dim=0)
        torch.save(iv["embeddings"], cache_path)

    return interviews


def assert_no_leakage(train, dev, test):
    """Hard guarantee: official splits are subject-disjoint."""
    tr = {iv["interview_id"] for iv in train}
    dv = {iv["interview_id"] for iv in dev}
    te = {iv["interview_id"] for iv in test}
    assert tr.isdisjoint(dv), "LEAKAGE: train ∩ dev"
    assert tr.isdisjoint(te), "LEAKAGE: train ∩ test"
    assert dv.isdisjoint(te), "LEAKAGE: dev ∩ test"
