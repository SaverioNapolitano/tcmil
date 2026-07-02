"""Raw DAIC-WOZ transcript preprocessing for dialogue-pair based depression detection.

This module processes the original tab-separated DAIC-WOZ transcripts and creates
Question-Answer (Q-A) dialogue pairs that preserve the conversational context.

Key insight: Embedding participant utterances in isolation loses critical context about
what question is being answered. "good" means very different things when responding to
"how are you today?" vs "how easy is it for you to get a good night's sleep?".

By pairing each interviewer question with the participant's response, the sentence
encoder receives the full conversational context needed to extract depression-relevant
features.

Processing pipeline:
    1. Parse raw tab-separated transcripts (start_time, stop_time, speaker, value)
    2. Merge consecutive same-speaker turns (handle split utterances)
    3. Create Q-A dialogue pairs in chronological order
    4. Clean text (remove non-verbal markers, normalize whitespace)
    5. Extract separate interviewer utterances for cross-role attention
"""

import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

# Pattern to match non-verbal markers like <clears throat>, <laughs>, etc.
_NONVERBAL_PATTERN = re.compile(r"<[^>]+>")

# Pattern to collapse multiple whitespace
_MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_utterance(text: str) -> str:
    """Clean a raw utterance text.

    Removes non-verbal markers (e.g., <clears throat>) but preserves
    disfluency markers (um, uh, etc.) as they are clinically relevant
    depression indicators.

    Args:
        text: Raw utterance text.

    Returns:
        Cleaned text string.
    """
    text = str(text).strip()

    # Remove non-verbal markers
    text = _NONVERBAL_PATTERN.sub("", text)

    # Collapse multiple spaces
    text = _MULTISPACE_PATTERN.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Transcript Parsing
# ---------------------------------------------------------------------------

def parse_raw_transcript(transcript_path: str | Path) -> list[dict]:
    """Parse a raw DAIC-WOZ transcript file into a list of turns.

    Args:
        transcript_path: Path to the raw tab-separated transcript CSV.

    Returns:
        List of turn dicts with keys: start_time, stop_time, speaker, value.
        Sorted by start_time (chronological order).
    """
    transcript_path = Path(transcript_path)

    df = pd.read_csv(transcript_path, sep="\t", on_bad_lines="skip")

    # Ensure required columns exist
    required_cols = {"start_time", "stop_time", "speaker", "value"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Transcript {transcript_path} missing columns. "
            f"Expected {required_cols}, got {set(df.columns)}"
        )

    # Sort by start_time for chronological order
    df = df.sort_values("start_time").reset_index(drop=True)

    turns = []
    for _, row in df.iterrows():
        text = clean_utterance(row["value"])
        if text and text.lower() != "nan":
            turns.append({
                "start_time": float(row["start_time"]),
                "stop_time": float(row["stop_time"]),
                "speaker": str(row["speaker"]).strip(),
                "value": text,
            })

    return turns


def merge_consecutive_turns(turns: list[dict]) -> list[dict]:
    """Merge consecutive turns by the same speaker into single logical utterances.

    Raw DAIC-WOZ transcripts often split one logical utterance across multiple
    rows (e.g., a participant pauses mid-sentence). This function merges them
    to produce cleaner semantic units.

    Args:
        turns: List of turn dicts sorted chronologically.

    Returns:
        List of merged turn dicts.
    """
    if not turns:
        return []

    merged = [turns[0].copy()]

    for turn in turns[1:]:
        prev = merged[-1]
        if turn["speaker"] == prev["speaker"]:
            # Same speaker: merge text and extend stop_time
            prev["value"] = prev["value"] + " " + turn["value"]
            prev["stop_time"] = max(prev["stop_time"], turn["stop_time"])
        else:
            merged.append(turn.copy())

    return merged


# ---------------------------------------------------------------------------
# Q-A Pair Creation
# ---------------------------------------------------------------------------

def create_dialogue_pairs(
    turns: list[dict],
    qa_separator: str = " [SEP] ",
    question_prefix: str = "[Q] ",
) -> tuple[list[str], list[str]]:
    """Create Question-Answer dialogue pairs from merged turns.

    Each Ellie question/prompt is paired with the subsequent Participant
    response(s). If a Participant speaks without a preceding Ellie turn
    (e.g., at the start), the response is kept with an empty question context.

    Args:
        turns: List of merged turn dicts in chronological order.
        qa_separator: Separator between question and answer text.
        question_prefix: Prefix for the question part.

    Returns:
        Tuple of:
        - qa_pairs: List of Q-A pair strings for patient embedding
        - interviewer_utterances: List of Ellie-only utterances for cross-role attention
    """
    qa_pairs = []
    interviewer_utterances = []

    i = 0
    pending_question = None

    while i < len(turns):
        turn = turns[i]

        if turn["speaker"] == "Ellie":
            # Collect interviewer utterance
            interviewer_utterances.append(turn["value"])

            # Set as pending question for the next participant response
            pending_question = turn["value"]
            i += 1

        elif turn["speaker"] == "Participant":
            participant_text = turn["value"]

            # Create Q-A pair
            if pending_question:
                qa_text = (
                    question_prefix + pending_question
                    + qa_separator + participant_text
                )
            else:
                # Participant speaks without a preceding question
                qa_text = participant_text

            qa_pairs.append(qa_text)
            pending_question = None
            i += 1

        else:
            # Unknown speaker, skip
            i += 1

    return qa_pairs, interviewer_utterances


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def process_transcript(
    transcript_path: str | Path,
) -> tuple[list[str], list[str], list[str]]:
    """Full preprocessing pipeline for a single transcript.

    Args:
        transcript_path: Path to the raw tab-separated transcript CSV.

    Returns:
        Tuple of:
        - qa_pairs: List of Q-A dialogue pair strings
        - interviewer_utterances: List of Ellie-only utterances
        - participant_utterances: List of participant-only utterances (for backward compat)
    """
    turns = parse_raw_transcript(transcript_path)
    merged = merge_consecutive_turns(turns)

    qa_pairs, interviewer_utterances = create_dialogue_pairs(merged)

    # Also extract participant-only utterances for backward compatibility
    participant_utterances = [
        t["value"] for t in merged if t["speaker"] == "Participant"
    ]

    return qa_pairs, interviewer_utterances, participant_utterances


def merge_by_gap(
    turns: list[dict],
    gap_threshold: float = 2.0,
) -> list[dict]:
    """Merge consecutive participant ASR segments into utterance instances.

    E-DAIC transcripts are un-diarized ASR output: every row is a short
    participant fragment (~1-3 s) with no speaker turns, so the speaker-based
    ``merge_consecutive_turns`` would collapse the whole session into a single
    instance. Instead we group fragments into utterance-like instances using
    silence: a gap >= ``gap_threshold`` seconds between one segment's stop_time
    and the next segment's start_time starts a new instance.

    Args:
        turns: List of turn dicts sorted chronologically.
        gap_threshold: Minimum silence (seconds) that splits two instances.

    Returns:
        List of merged turn dicts (one per utterance instance).
    """
    if not turns:
        return []

    merged = [turns[0].copy()]
    for turn in turns[1:]:
        prev = merged[-1]
        if turn["start_time"] - prev["stop_time"] < gap_threshold:
            prev["value"] = prev["value"] + " " + turn["value"]
            prev["stop_time"] = max(prev["stop_time"], turn["stop_time"])
        else:
            merged.append(turn.copy())
    return merged


def process_edaic_transcript(
    transcript_path: str | Path,
    gap_threshold: float = 2.0,
) -> list[str]:
    """Preprocess a converted E-DAIC transcript into participant-only instances.

    E-DAIC has no interviewer (Ellie) turns, so the Q-A pairing used for
    DAIC-WOZ does not apply. This produces participant-only utterance instances
    via gap-based segmentation, for the zero-shot generalization experiment
    (TCMIL trained on DAIC-WOZ, tested on E-DAIC). For a matched comparison, run
    DAIC-WOZ through the participant-only path as well (see process_transcript's
    participant_utterances output).

    Args:
        transcript_path: Path to a converted E-DAIC transcript
            (``{ID}_TRANSCRIPT.csv``, DAIC-WOZ TSV schema, speaker=Participant).
        gap_threshold: Silence (seconds) that splits two utterance instances.

    Returns:
        List of cleaned participant utterance instance strings (the MIL bag).
    """
    turns = parse_raw_transcript(transcript_path)
    turns = [t for t in turns if t["speaker"] == "Participant"]
    instances = merge_by_gap(turns, gap_threshold=gap_threshold)
    return [t["value"] for t in instances if t["value"]]


def process_all_transcripts(
    data_dir: str | Path,
    participant_ids: list[int],
) -> dict[int, dict]:
    """Process all raw transcripts for given participant IDs.

    Args:
        data_dir: Root data directory containing 'raw/' subdirectory.
        participant_ids: List of participant IDs to process.

    Returns:
        Dict mapping participant_id -> {qa_pairs, interviewer_utterances, participant_utterances}
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"

    results = {}
    for pid in participant_ids:
        transcript_path = raw_dir / f"{pid}_TRANSCRIPT.csv"
        if not transcript_path.exists():
            print(f"  [WARN] No raw transcript for participant {pid}, skipping.")
            continue

        try:
            qa_pairs, interviewer_utts, participant_utts = process_transcript(
                transcript_path
            )
            results[pid] = {
                "qa_pairs": qa_pairs,
                "interviewer_utterances": interviewer_utts,
                "participant_utterances": participant_utts,
            }
        except Exception as e:
            print(f"  [WARN] Failed to process transcript {pid}: {e}")

    return results
