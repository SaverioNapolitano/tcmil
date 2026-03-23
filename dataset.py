"""Dataset loader for DAIC-WOZ interview transcripts with binary depression labels."""

from pathlib import Path

import numpy as np
import pandas as pd


# Column name for binary label differs between train/dev and test splits
_LABEL_COL = {"train": "PHQ8_Binary", "dev": "PHQ8_Binary", "test": "PHQ_Binary"}


def load_interviews(
    data_dir: str | Path,
    split: str,
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
    transcript_dir = data_dir / "preprocessed" / split
    interviews = []

    for pid, label in sorted(id_to_label.items()):
        transcript_path = transcript_dir / f"{pid}_TRANSCRIPT.csv"

        if not transcript_path.exists():
            print(f"  [WARN] No transcript file for participant {pid}, skipping.")
            continue

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


def load_all_interviews(data_dir: str | Path) -> list[dict]:
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
            interviews = load_interviews(data_dir, split)
            all_interviews.extend(interviews)
        except Exception as e:
            print(f"Warning: Could not load split '{split}': {e}")
            
    print(f"\nLoaded {len(all_interviews)} total interviews across all splits.")
    return all_interviews
