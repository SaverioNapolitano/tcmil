"""Dataset loader for DAIC-WOZ interview transcripts with binary depression labels."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Column name for binary label differs between train/dev and test splits
_LABEL_COL = {"train": "PHQ8_Binary", "dev": "PHQ8_Binary", "test": "PHQ_Binary"}

# PHQ-8 individual symptom components (only present in train/dev)
_SYMPTOM_COLS = [
    "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
    "PHQ8_Appetite", "PHQ8_Failure", "PHQ8_Concentrating", "PHQ8_Moving"
]


def load_interviews(
    data_dir: str | Path,
    split: str,
    use_raw_data: bool = False,
) -> list[dict]:
    """Load all interviews for a given split.

    Each interview is returned as a dict with keys:
        - interview_id: int (Participant_ID)
        - label: int (0 or 1)
        - utterances: list[str] (participant utterances only, non-empty)

    Ellie's utterances are filtered out — this baseline uses only
    participant speech. Empty or whitespace-only utterances are skipped.

    Args:
        data_dir: Root data directory (e.g., "data").
        split: One of "train", "dev", "test".

    Returns:
        List of interview dictionaries.
    """
    data_dir = Path(data_dir)

    # --- Load labels ---
    label_files = {
        "train": "train_split_Depression_AVEC2017.csv",
        "dev": "dev_split_Depression_AVEC2017.csv",
        "test": "full_test_split.csv",
    }
    label_path = data_dir / "labels" / label_files[split]
    labels_df = pd.read_csv(label_path)
    label_col = _LABEL_COL[split]

    # Build {participant_id: label} mapping
    id_to_label = dict(
        zip(labels_df["Participant_ID"], labels_df[label_col])
    )

    # --- Load transcripts ---
    if use_raw_data:
        transcript_dir = data_dir / "raw"
    else:
        transcript_dir = data_dir / "preprocessed" / split
    interviews = []

    for pid, label in sorted(id_to_label.items()):
        transcript_path = transcript_dir / f"{pid}_TRANSCRIPT.csv"

        if not transcript_path.exists():
            print(f"  [WARN] No transcript file for participant {pid}, skipping.")
            continue

        if use_raw_data:
            df = pd.read_csv(transcript_path, sep='\t', on_bad_lines='skip')
        else:
            df = pd.read_csv(transcript_path)

        # Keep only participant utterances (ignore Ellie / interviewer)
        participant_df = df[df["speaker"] == "Participant"]

        # Extract non-empty utterance texts
        utterances = []
        for text in participant_df["value"]:
            text = str(text).strip()
            if text and text.lower() != "nan":
                utterances.append(text)

        interviews.append({
            "interview_id": int(pid),
            "label": int(label),
            "utterances": utterances,
        })

    return interviews


def print_split_stats(interviews: list[dict], split: str) -> list[int]:
    """Print summary statistics for a split and return utterance counts.

    Args:
        interviews: List of interview dicts from load_interviews.
        split: Split name for display.

    Returns:
        List of utterance counts per interview.
    """
    n = len(interviews)
    labels = [iv["label"] for iv in interviews]
    counts = [len(iv["utterances"]) for iv in interviews]

    n_pos = sum(labels)
    n_neg = n - n_pos

    print(f"\n--- {split.upper()} split ---")
    print(f"  Interviews: {n}")
    print(f"  Positive (depressed): {n_pos} ({100*n_pos/n:.1f}%)")
    print(f"  Negative (not depressed): {n_neg} ({100*n_neg/n:.1f}%)")
    print(f"  Utterances per interview:")
    print(f"    mean={np.mean(counts):.1f}, std={np.std(counts):.1f}")
    print(f"    min={np.min(counts)}, max={np.max(counts)}")

    zero_utt = sum(1 for c in counts if c == 0)
    if zero_utt > 0:
        print(f"  [WARN] {zero_utt} interviews with ZERO utterances!")

    return counts


def load_all_interviews(data_dir: str | Path, use_raw_data: bool = False) -> list[dict]:
    """Load and combine train, dev, and test splits into a single list.
    
    Useful for cross-validation where new splits are generated dynamically.
    
    Args:
        data_dir: Path to the data directory.
        
    Returns:
        A list of interview dictionaries containing all available data.
    """
    all_interviews = []
    for split in ["train", "dev", "test"]:
        try:
            interviews = load_interviews(data_dir, split, use_raw_data=use_raw_data)
            all_interviews.extend(interviews)
        except Exception as e:
            print(f"Warning: Could not load split '{split}': {e}")
            
    print(f"\nLoaded {len(all_interviews)} total interviews across all splits.")
    return all_interviews


def load_interviews_with_roles(
    data_dir: str | Path,
    split: str,
    use_raw_data: bool = False,
) -> list[dict]:
    """Load all interviews for a given split, keeping both participant and interviewer utterances.

    Each interview is returned as a dict with keys:
        - interview_id: int (Participant_ID)
        - label: int (0 or 1)
        - utterances: list[str] (participant utterances, non-empty)
        - interviewer_utterances: list[str] (Ellie utterances, non-empty)

    Args:
        data_dir: Root data directory (e.g., "data").
        split: One of "train", "dev", "test".

    Returns:
        List of interview dictionaries with both roles.
    """
    data_dir = Path(data_dir)

    # --- Load labels ---
    label_files = {
        "train": "train_split_Depression_AVEC2017.csv",
        "dev": "dev_split_Depression_AVEC2017.csv",
        "test": "full_test_split.csv",
    }
    label_path = data_dir / "labels" / label_files[split]
    labels_df = pd.read_csv(label_path)
    label_col = _LABEL_COL[split]

    # Build {participant_id: label} mapping
    id_to_label = dict(
        zip(labels_df["Participant_ID"], labels_df[label_col])
    )

    # --- Load transcripts ---
    if use_raw_data:
        transcript_dir = data_dir / "raw"
    else:
        transcript_dir = data_dir / "preprocessed" / split
    interviews = []

    for pid, label in sorted(id_to_label.items()):
        transcript_path = transcript_dir / f"{pid}_TRANSCRIPT.csv"

        if not transcript_path.exists():
            print(f"  [WARN] No transcript file for participant {pid}, skipping.")
            continue

        if use_raw_data:
            df = pd.read_csv(transcript_path, sep='\t', on_bad_lines='skip')
        else:
            df = pd.read_csv(transcript_path)

        # Participant utterances
        participant_df = df[df["speaker"] == "Participant"]
        utterances = []
        for text in participant_df["value"]:
            text = str(text).strip()
            if text and text.lower() != "nan":
                utterances.append(text)

        # Interviewer (Ellie) utterances
        ellie_df = df[df["speaker"] == "Ellie"]
        interviewer_utterances = []
        for text in ellie_df["value"]:
            text = str(text).strip()
            if text and text.lower() != "nan":
                interviewer_utterances.append(text)

        # Symptom indicators (0-3 Ordinal)
        symptoms = []
        has_symptoms = all(col in labels_df.columns for col in _SYMPTOM_COLS)
        if has_symptoms:
            row = labels_df[labels_df["Participant_ID"] == pid].iloc[0]
            for col in _SYMPTOM_COLS:
                val = row[col]
                if pd.isna(val):
                    has_symptoms = False
                    break
                symptoms.append(float(val))
            
            if not has_symptoms:
                symptoms = [0.0] * len(_SYMPTOM_COLS)
        else:
            symptoms = [0.0] * len(_SYMPTOM_COLS)

        interviews.append({
            "interview_id": int(pid),
            "label": int(label),
            "utterances": utterances,
            "interviewer_utterances": interviewer_utterances,
            "symptoms": symptoms,
            "has_symptoms": has_symptoms,
        })

    return interviews


def load_all_interviews_with_roles(data_dir: str | Path, use_raw_data: bool = False) -> list[dict]:
    """Load and combine all splits with both participant and interviewer utterances.

    Useful for cross-validation with DAMIL-R where both roles are needed.

    Args:
        data_dir: Path to the data directory.

    Returns:
        A list of interview dictionaries with both roles across all splits.
    """
    all_interviews = []
    for split in ["train", "dev", "test"]:
        try:
            interviews = load_interviews_with_roles(data_dir, split, use_raw_data=use_raw_data)
            all_interviews.extend(interviews)
        except Exception as e:
            print(f"Warning: Could not load split '{split}': {e}")

    print(f"\nLoaded {len(all_interviews)} total interviews (with roles) across all splits.")
    return all_interviews


def load_all_interviews_dialogue_pairs(data_dir: str | Path) -> list[dict]:
    """Load all interviews with Q-A dialogue pairs from raw transcripts.

    Uses preprocess_raw.py to parse raw DAIC-WOZ transcripts, merge consecutive
    same-speaker turns, and create Question-Answer dialogue pairs. This gives the
    sentence encoder crucial context about what topic each participant response
    addresses.

    Args:
        data_dir: Path to the data directory (must contain 'raw/' and 'labels/' subdirectories).

    Returns:
        A list of interview dictionaries with keys:
        - interview_id: int
        - label: int (0 or 1)
        - qa_pairs: list[str] (Q-A dialogue pair strings)
        - interviewer_utterances: list[str] (Ellie-only utterances)
        - utterances: list[str] (participant-only utterances, for backward compat)
        - symptoms: list[float] (8 PHQ-8 symptom scores)
        - has_symptoms: bool
    """
    from preprocess_raw import process_all_transcripts

    data_dir = Path(data_dir)

    # Collect all participant IDs and labels across splits
    label_files = {
        "train": ("train_split_Depression_AVEC2017.csv", "PHQ8_Binary"),
        "dev": ("dev_split_Depression_AVEC2017.csv", "PHQ8_Binary"),
        "test": ("full_test_split.csv", "PHQ_Binary"),
    }

    id_to_label = {}
    all_label_dfs = {}

    for split, (filename, label_col) in label_files.items():
        label_path = data_dir / "labels" / filename
        try:
            df = pd.read_csv(label_path)
            all_label_dfs[split] = df
            for _, row in df.iterrows():
                pid = int(row["Participant_ID"])
                label = int(row[label_col])
                id_to_label[pid] = label
        except Exception as e:
            print(f"Warning: Could not load labels for '{split}': {e}")

    # Process all raw transcripts
    all_pids = sorted(id_to_label.keys())
    print(f"Processing {len(all_pids)} raw transcripts for dialogue pairs...")
    processed = process_all_transcripts(data_dir, all_pids)

    # Build interview list with symptom info
    interviews = []
    for pid in all_pids:
        if pid not in processed:
            continue

        proc = processed[pid]
        label = id_to_label[pid]

        # Get symptom info from the train/dev label files (test doesn't have symptoms)
        symptoms = [0.0] * len(_SYMPTOM_COLS)
        has_symptoms = False

        for split, df in all_label_dfs.items():
            if split == "test":
                continue
            match = df[df["Participant_ID"] == pid]
            if len(match) > 0 and all(col in df.columns for col in _SYMPTOM_COLS):
                row = match.iloc[0]
                sym_vals = []
                valid = True
                for col in _SYMPTOM_COLS:
                    val = row[col]
                    if pd.isna(val):
                        valid = False
                        break
                    sym_vals.append(float(val))
                if valid:
                    symptoms = sym_vals
                    has_symptoms = True
                break

        interviews.append({
            "interview_id": pid,
            "label": label,
            "qa_pairs": proc["qa_pairs"],
            "interviewer_utterances": proc["interviewer_utterances"],
            "utterances": proc["participant_utterances"],
            "symptoms": symptoms,
            "has_symptoms": has_symptoms,
        })

    print(f"\nLoaded {len(interviews)} total interviews with dialogue pairs.")
    return interviews


# ---------------------------------------------------------------------------
# Text Augmentation (for on-the-fly re-embedding)
# ---------------------------------------------------------------------------

import random as _random


def augment_utterances_word_dropout(
    utterances: list[str], drop_rate: float = 0.05
) -> list[str]:
    """Drop words independently from each utterance.

    Simulates disfluencies and partial responses common in clinical
    interviews.  Preserves at least 2 words per utterance.
    """
    augmented = []
    for u in utterances:
        words = u.split()
        if len(words) > 2 and drop_rate > 0:
            kept = [w for w in words if _random.random() > drop_rate]
            if len(kept) < 2:
                kept = words[:2]
            augmented.append(" ".join(kept))
        else:
            augmented.append(u)
    return augmented


def augment_utterances_deletion(
    utterances: list[str], drop_rate: float = 0.2
) -> list[str]:
    """Randomly remove entire utterances from a bag.

    Produces genuinely different embedding bags when re-encoded.
    Preserves at least 3 utterances.
    """
    if len(utterances) <= 3 or drop_rate <= 0:
        return list(utterances)
    kept = [u for u in utterances if _random.random() > drop_rate]
    if len(kept) < 3:
        kept = list(utterances)[:3]
    return kept


def create_augmented_interview(
    interview: dict,
    aug_idx: int,
    word_drop_rate: float = 0.05,
    utt_drop_rate: float = 0.2,
) -> dict:
    """Return an augmented copy of *interview* with modified patient text.

    The copy has the same label, symptoms, and interviewer utterances but
    different patient utterances (after word dropout + utterance deletion).
    Pre-computed embeddings are **removed** — the caller must re-embed.
    """
    aug = {**interview}
    aug["interview_id"] = f"{interview['interview_id']}_aug{aug_idx}"

    utts = list(interview["utterances"])
    utts = augment_utterances_deletion(utts, drop_rate=utt_drop_rate)
    utts = augment_utterances_word_dropout(utts, drop_rate=word_drop_rate)
    aug["utterances"] = utts

    # Remove stale embeddings — they must be re-computed from augmented text
    aug.pop("patient_embeddings", None)
    aug.pop("interviewer_embeddings", None)
    return aug


# ---------------------------------------------------------------------------
# Datasets and Collation
# ---------------------------------------------------------------------------

class DualRoleBagDataset(Dataset):
    """Dataset for pre-computed utterance embeddings with both roles.

    Supports instance dropout: during training, randomly drops utterances
    from each bag to create diverse views of each interview.
    """

    def __init__(self, interviews: list[dict], instance_dropout: float = 0.0):
        self.interviews = interviews
        self.instance_dropout = instance_dropout

    def __len__(self):
        return len(self.interviews)

    def __getitem__(self, idx):
        item = self.interviews[idx]
        patient_bag = item["patient_embeddings"]       # (P, d)
        interviewer_bag = item["interviewer_embeddings"]  # (I, d)
        patient_ling = item.get("patient_ling_features", None) # (P, 16)

        # Instance dropout: randomly drop utterances during training
        if self.instance_dropout > 0:
            if patient_bag.size(0) > 2:
                mask = torch.rand(patient_bag.size(0)) > self.instance_dropout
                mask[0] = True
                if mask.sum() < 2: mask[:2] = True
                patient_bag = patient_bag[mask]
                if patient_ling is not None:
                    patient_ling = patient_ling[mask]

            if interviewer_bag.size(0) > 2:
                mask = torch.rand(interviewer_bag.size(0)) > self.instance_dropout
                mask[0] = True
                if mask.sum() < 2: mask[:2] = True
                interviewer_bag = interviewer_bag[mask]

        return {
            "patient_bag": patient_bag,
            "interviewer_bag": interviewer_bag,
            "label": torch.tensor(item["label"], dtype=torch.float),
            "interview_id": item["interview_id"],
            "symptoms": torch.tensor(item.get("symptoms", [0]*8), dtype=torch.float),
            "has_symptoms": torch.tensor(1.0 if item.get("has_symptoms", False) else 0.0, dtype=torch.float),
            "utterances": item.get("utterances", []),
            "interviewer_utterances": item.get("interviewer_utterances", []),
            "patient_ling": patient_ling,
        }


def collate_dual_role_bags(batch):
    """Collate pre-computed embeddings into padded batches."""
    patient_bags = [item["patient_bag"] for item in batch]
    interviewer_bags = [item["interviewer_bag"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    ids = [item["interview_id"] for item in batch]
    utts = [item["utterances"] for item in batch]
    int_utts = [item["interviewer_utterances"] for item in batch]
    patient_lings = [item.get("patient_ling") for item in batch]
    has_ling = all(l is not None for l in patient_lings)

    patient_sizes = [bag.size(0) for bag in patient_bags]
    interviewer_sizes = [bag.size(0) for bag in interviewer_bags]

    max_p = max(patient_sizes)
    max_i = max(interviewer_sizes)
    d = patient_bags[0].size(1)

    padded_patient = torch.zeros(len(batch), max_p, d)
    padded_interviewer = torch.zeros(len(batch), max_i, d)
    
    padded_ling = None
    if has_ling:
        d_ling = patient_lings[0].size(1)
        padded_ling = torch.zeros(len(batch), max_p, d_ling)

    for idx, (p_bag, i_bag) in enumerate(zip(patient_bags, interviewer_bags)):
        padded_patient[idx, :patient_sizes[idx], :] = p_bag
        padded_interviewer[idx, :interviewer_sizes[idx], :] = i_bag
        if has_ling:
            padded_ling[idx, :patient_sizes[idx], :] = patient_lings[idx]

    return {
        "patient_bags": padded_patient,
        "interviewer_bags": padded_interviewer,
        "patient_sizes": patient_sizes,
        "interviewer_sizes": interviewer_sizes,
        "labels": labels,
        "symptoms": torch.stack([item["symptoms"] for item in batch]),
        "has_symptoms": torch.stack([item["has_symptoms"] for item in batch]),
        "interview_ids": ids,
        "utterances_lists": utts,
        "interviewer_utterances_lists": int_utts,
        "patient_ling_features": padded_ling,
    }


class TokenizedDualRoleBagDataset(Dataset):
    """Dataset for raw text utterances to be tokenized on-the-fly (for LoRA)."""

    def __init__(self, interviews: list[dict], instance_dropout: float = 0.0):
        self.interviews = interviews
        self.instance_dropout = instance_dropout

    def __len__(self):
        return len(self.interviews)

    def __getitem__(self, idx):
        item = self.interviews[idx]
        p_utts = item["utterances"]
        i_utts = item["interviewer_utterances"]

        # Instance dropout on text level
        if self.instance_dropout > 0:
            if len(p_utts) > 2:
                p_utts = [u for u in p_utts if random.random() > self.instance_dropout]
                if len(p_utts) < 2: p_utts = item["utterances"][:2]
            if len(i_utts) > 2:
                i_utts = [u for u in i_utts if random.random() > self.instance_dropout]
                if len(i_utts) < 2: i_utts = item["interviewer_utterances"][:2]

        return {
            "patient_utterances": p_utts,
            "interviewer_utterances": i_utts,
            "label": torch.tensor(item["label"], dtype=torch.float),
            "interview_id": item["interview_id"],
        }


def collate_lora_bags(batch, tokenizer, max_len=128):
    """Collates raw text into token IDs for Transformer fine-tuning (LoRA)."""
    # Note: Only works with batch_size=1 due to MIL turn-level variability
    assert len(batch) == 1, "LoRA MIL collator currently supports batch_size=1 only."
    item = batch[0]
    
    p_encoded = tokenizer(
        item["patient_utterances"], 
        padding=True, 
        truncation=True, 
        max_length=max_len, 
        return_tensors="pt"
    )
    
    i_encoded = tokenizer(
        item["interviewer_utterances"], 
        padding=True, 
        truncation=True, 
        max_length=max_len, 
        return_tensors="pt"
    )

    return {
        "patient_bags": p_encoded,        # {'input_ids': (P, L), 'attention_mask': (P, L)}
        "interviewer_bags": i_encoded,    # {'input_ids': (I, L), 'attention_mask': (I, L)}
        "labels": item["label"].unsqueeze(0),
        "interview_ids": [item["interview_id"]],
    }
