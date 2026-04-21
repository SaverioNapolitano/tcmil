"""Per-utterance clinical linguistic feature extraction for DAIC-WOZ participants.

Computes 16 features per participant utterance using pure Python (no external NLP
dependencies). All features are grounded in clinical NLP literature on depression:

  - Disfluency & filler rates  : speech production difficulty
  - First-person singular rate : self-focused cognition (e.g., Rude et al., 2004)
  - Negation rate              : negativity bias
  - Hedge rate                 : uncertainty / reduced confidence
  - Negative/positive affect   : lexicon-based sentiment proxy
  - Past-tense rate            : rumination on past events
  - Brevity features           : psychomotor retardation marker
  - Turn position              : temporal location in the interview
  - Relative length            : deviation from subject's own baseline

Raw (unnormalized) features are returned. Callers should fit a StandardScaler on
the training split and transform all splits — see `normalize_ling_features()`.
"""

import re
from typing import Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Feature index names
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "word_count_log",       # 0  log(1 + n_words)
    "type_token_ratio",     # 1  |unique words| / n_words
    "disfluency_rate",      # 2  (um + uh + hmm + ...) / n_words
    "filler_rate",          # 3  single-word fillers + phrase fillers / n_words
    "first_sg_rate",        # 4  I / me / my / mine / myself / n_words
    "first_pl_rate",        # 5  we / us / our / ours / ourselves / n_words
    "negation_rate",        # 6  no / not / never / nothing / n't / ... / n_words
    "hedge_rate",           # 7  maybe / perhaps / might / could / ... / n_words
    "neg_affect_rate",      # 8  negative-affect lexicon count / n_words
    "pos_affect_rate",      # 9  positive-affect lexicon count / n_words
    "past_tense_rate",      # 10 past-tense verb proxies / n_words
    "question_marker",      # 11 binary: utterance ends with '?'
    "avg_word_len_norm",    # 12 mean character length per word / 10
    "sentence_count_log",   # 13 log(1 + n_sentences)
    "turn_position",        # 14 utterance index / (n_utterances - 1)  ∈ [0, 1]
    "relative_length",      # 15 (n_words - μ_subject) / σ_subject, clipped ±3
]

N_FEATURES: int = len(FEATURE_NAMES)  # 16

# ---------------------------------------------------------------------------
# Lexicons (all lower-cased)
# ---------------------------------------------------------------------------

_DISFLUENCIES = frozenset({
    "um", "uh", "mm", "hmm", "hm", "er", "ah", "uhh", "umm",
})

_FILLERS_WORD = frozenset({"like"})           # word-level
_FILLERS_PHRASE = ("you know", "i mean", "sort of", "kind of", "i guess")

_FIRST_SG = frozenset({"i", "me", "my", "mine", "myself"})
_FIRST_PL = frozenset({"we", "us", "our", "ours", "ourselves"})

_NEGATIONS = frozenset({
    "no", "not", "never", "nothing", "nobody", "nowhere", "neither", "nor",
    "without", "lack", "lacking", "unable", "cannot",
})

_HEDGES = frozenset({
    "maybe", "perhaps", "probably", "might", "could", "possibly", "sometimes",
    "usually", "often", "generally", "somewhat", "fairly", "rather", "quite",
    "think", "guess", "suppose", "seem", "seems", "seemed",
})

_NEG_AFFECT = frozenset({
    "sad", "depressed", "unhappy", "miserable", "terrible", "awful", "horrible",
    "bad", "wrong", "difficult", "hard", "problem", "struggle", "pain",
    "hurt", "angry", "frustrated", "tired", "exhausted", "worthless", "hopeless",
    "useless", "guilty", "ashamed", "lonely", "alone", "empty", "lost", "hate",
    "fear", "afraid", "anxious", "worry", "worried", "stressed", "stress",
    "nervous", "confused", "dark", "death", "die", "dying", "sick", "ill",
    "weak", "fail", "failed", "failure", "broke", "broken", "crying", "cry",
    "tears", "suffer", "suffering", "misery", "grief", "helpless", "desperate",
})

_POS_AFFECT = frozenset({
    "happy", "good", "great", "wonderful", "fine", "okay", "well", "better",
    "nice", "love", "enjoy", "glad", "excited", "fun", "positive", "pleased",
    "delighted", "grateful", "thankful", "hope", "hopeful", "strong", "confident",
    "peaceful", "calm", "relaxed", "comfortable", "content", "satisfied", "joy",
    "laugh", "smile", "proud", "achieve", "achieved", "success", "support",
    "care", "caring", "kind", "warm", "safe", "healthy", "energetic",
})

_PAST_TENSE = frozenset({
    "was", "were", "had", "did", "went", "used", "felt", "thought", "knew",
    "said", "told", "saw", "came", "got", "lost", "found", "made", "took",
    "kept", "left", "gave", "put", "seemed", "became", "happened", "tried",
    "started", "stopped", "wanted", "needed", "called", "worked", "lived",
    "moved", "changed", "ended", "began", "realized", "noticed", "remembered",
})

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_TOKENIZE_RE = re.compile(r"\b\w+\b")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _tokenize(text: str) -> list[str]:
    return _TOKENIZE_RE.findall(text.lower())


def _count_phrase(text_lower: str, phrase: str) -> int:
    return text_lower.count(phrase)


# ---------------------------------------------------------------------------
# Single-utterance feature extraction
# ---------------------------------------------------------------------------

def _extract_utterance_features(
    text: str,
    turn_idx: int,
    n_turns: int,
    subject_word_counts: np.ndarray,
) -> np.ndarray:
    """Return the 16-dimensional feature vector for a single utterance.

    Args:
        text: Raw utterance string.
        turn_idx: Zero-based index of this utterance within the interview.
        n_turns: Total number of participant utterances in the interview.
        subject_word_counts: Array of word counts for ALL utterances of this
            subject (used to compute the intra-subject z-score).

    Returns:
        Float32 array of shape (N_FEATURES,).
    """
    text_lower = text.lower()
    words = _tokenize(text)
    n_words = max(len(words), 1)
    unique_words = set(words)
    word_count = len(words)

    # 0: word count (log-normalized)
    f0 = float(np.log1p(word_count))

    # 1: type-token ratio
    f1 = len(unique_words) / n_words

    # 2: disfluency rate
    f2 = sum(1 for w in words if w in _DISFLUENCIES) / n_words

    # 3: filler rate (word-level + phrase-level)
    filler_cnt = sum(1 for w in words if w in _FILLERS_WORD)
    filler_cnt += sum(_count_phrase(text_lower, ph) for ph in _FILLERS_PHRASE)
    f3 = filler_cnt / n_words

    # 4: first-person singular
    f4 = sum(1 for w in words if w in _FIRST_SG) / n_words

    # 5: first-person plural
    f5 = sum(1 for w in words if w in _FIRST_PL) / n_words

    # 6: negation rate (word + contraction "n't")
    neg_cnt = sum(1 for w in words if w in _NEGATIONS)
    neg_cnt += _count_phrase(text_lower, "n't")
    f6 = neg_cnt / n_words

    # 7: hedge rate
    f7 = sum(1 for w in words if w in _HEDGES) / n_words

    # 8: negative affect
    f8 = sum(1 for w in words if w in _NEG_AFFECT) / n_words

    # 9: positive affect
    f9 = sum(1 for w in words if w in _POS_AFFECT) / n_words

    # 10: past-tense rate
    f10 = sum(1 for w in words if w in _PAST_TENSE) / n_words

    # 11: question marker (binary)
    f11 = 1.0 if "?" in text else 0.0

    # 12: average word length (normalized by 10)
    f12 = (sum(len(w) for w in words) / n_words) / 10.0 if words else 0.0

    # 13: sentence count (log-normalized against max=10)
    n_sents = max(len([s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]), 1)
    f13 = float(np.log1p(n_sents)) / float(np.log1p(10))

    # 14: normalized turn position ∈ [0, 1]
    f14 = float(turn_idx) / float(max(n_turns - 1, 1))

    # 15: relative length (z-score within this subject, clipped to ±3)
    mean_wc = float(subject_word_counts.mean()) if len(subject_word_counts) > 1 else float(word_count)
    std_wc = float(subject_word_counts.std()) if len(subject_word_counts) > 1 else 1.0
    std_wc = max(std_wc, 1e-3)
    f15 = float(np.clip((word_count - mean_wc) / std_wc, -3.0, 3.0))

    return np.array(
        [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Interview-level extraction
# ---------------------------------------------------------------------------

def extract_interview_ling_features(interview: dict) -> np.ndarray:
    """Extract linguistic features for all participant utterances of one interview.

    Args:
        interview: Interview dict with at least the 'utterances' key.

    Returns:
        Float32 array of shape (P, N_FEATURES) where P = len(utterances).
        If the interview has no utterances, returns shape (1, N_FEATURES) of zeros.
    """
    utterances: list[str] = interview.get("utterances", [])
    if not utterances:
        return np.zeros((1, N_FEATURES), dtype=np.float32)

    n_turns = len(utterances)
    word_counts = np.array(
        [max(len(_tokenize(u)), 1) for u in utterances], dtype=np.float32
    )

    features = np.stack(
        [
            _extract_utterance_features(utt, i, n_turns, word_counts)
            for i, utt in enumerate(utterances)
        ]
    )  # (P, N_FEATURES)

    return features


def extract_all_ling_features(interviews: list[dict]) -> list[dict]:
    """Augment each interview with 'patient_ling_features' (raw, unnormalized).

    The returned list contains the same interviews with an additional key:
        ``patient_ling_features``: torch.Tensor of shape (P, N_FEATURES).

    No normalization is applied here. Use :func:`normalize_ling_features` inside
    the training loop, fitting the scaler only on the training split.

    Args:
        interviews: List of interview dicts from dataset loaders.

    Returns:
        New list of dicts (original dicts are not mutated).
    """
    result = []
    for iv in interviews:
        feats = extract_interview_ling_features(iv)
        result.append({
            **iv,
            "patient_ling_features": torch.tensor(feats, dtype=torch.float32),
        })
    return result


# ---------------------------------------------------------------------------
# Normalization (must be fit on training split only)
# ---------------------------------------------------------------------------

def normalize_ling_features(
    interviews: list[dict],
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    fit: bool = False,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Z-score normalize 'patient_ling_features' using per-feature statistics.

    Args:
        interviews: List of interview dicts with 'patient_ling_features' tensors.
        mean: Pre-fitted feature means of shape (N_FEATURES,). Ignored if fit=True.
        std: Pre-fitted feature stds of shape (N_FEATURES,). Ignored if fit=True.
        fit: If True, compute mean/std from this data (use on training split only).

    Returns:
        Tuple of:
        - New list of dicts with normalized 'patient_ling_features'.
        - mean array (computed or passed through).
        - std array (computed or passed through).
    """
    if fit:
        all_feats = np.concatenate(
            [iv["patient_ling_features"].numpy() for iv in interviews], axis=0
        )  # (total_utterances, N_FEATURES)
        mean = all_feats.mean(axis=0)
        std = all_feats.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)  # avoid division by zero

    assert mean is not None and std is not None, "Provide mean/std or set fit=True"

    result = []
    for iv in interviews:
        raw = iv["patient_ling_features"].numpy()
        normalized = ((raw - mean) / std).astype(np.float32)
        result.append({**iv, "patient_ling_features": torch.tensor(normalized)})

    return result, mean, std
