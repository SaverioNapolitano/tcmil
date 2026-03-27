"""DAMIL-H attention diagnostics and visualization.

Provides functions for:
- Attention weight histograms across all dialogues
- Per-dialogue attention heatmaps
- Top-attended utterance reports
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")


def plot_attention_histogram(
    all_weights: list[np.ndarray],
    split_name: str,
    output_dir: Path,
) -> None:
    """Plot histogram of all individual attention weights across dialogues.

    This shows how attention mass is distributed — peaky distributions
    indicate the model focuses on a few key utterances.

    Args:
        all_weights: List of attention weight arrays (one per dialogue).
        split_name: Split name for title and filename.
        output_dir: Directory to save the plot.
    """
    flat_weights = np.concatenate(all_weights)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw weight distribution
    axes[0].hist(flat_weights, bins=50, color="teal", alpha=0.7, edgecolor="white")
    axes[0].set_xlabel("Attention Weight")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Attention Weight Distribution — {split_name}")
    axes[0].axvline(x=np.mean(flat_weights), color="red", linestyle="--", label=f"mean={np.mean(flat_weights):.4f}")
    axes[0].legend()

    # Max weight per dialogue (concentration measure)
    max_weights = [w.max() for w in all_weights]
    axes[1].hist(max_weights, bins=30, color="coral", alpha=0.7, edgecolor="white")
    axes[1].set_xlabel("Max Attention Weight per Dialogue")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Attention Concentration — {split_name}")
    axes[1].axvline(x=np.mean(max_weights), color="red", linestyle="--", label=f"mean={np.mean(max_weights):.4f}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"attention_histogram_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_attention_dialogue(
    interview_id: int,
    attention_weights: np.ndarray,
    utterance_texts: list[str],
    true_label: int,
    predicted_prob: float,
    output_dir: Path,
    top_k: int = 10,
) -> None:
    """Visualize attention weights for a single dialogue.

    Produces a horizontal bar chart highlighting the most-attended utterances,
    with text labels for interpretability.

    Args:
        interview_id: Participant ID.
        attention_weights: Array of shape (num_utterances,).
        utterance_texts: Corresponding utterance strings.
        true_label: Ground truth label (0 or 1).
        predicted_prob: Model's predicted probability.
        output_dir: Directory to save the plot.
        top_k: Number of top-attended utterances to highlight.
    """
    n = len(attention_weights)
    indices = np.arange(n)

    # Sort by attention weight (descending) for the top-k highlight
    sorted_idx = np.argsort(attention_weights)[::-1]
    top_indices = set(sorted_idx[:top_k])

    # Color: top-k in coral, rest in grey
    colors = ["coral" if i in top_indices else "lightsteelblue" for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.25)))
    ax.barh(indices, attention_weights, color=colors, edgecolor="white", height=0.8)

    # Add truncated text labels
    for i in range(n):
        text = utterance_texts[i][:50] + ("..." if len(utterance_texts[i]) > 50 else "")
        ax.text(
            attention_weights[i] + 0.001, i, text,
            va="center", fontsize=7, color="grey",
        )

    ax.set_yticks(indices)
    ax.set_yticklabels([f"U{i}" for i in range(n)], fontsize=8)
    ax.set_xlabel("Attention Weight")
    ax.set_title(
        f"DAMIL-H Attention — Interview {interview_id}\n"
        f"True: {'Depressed' if true_label == 1 else 'Not Depressed'} | "
        f"Pred Prob: {predicted_prob:.3f}"
    )
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_dir / f"attention_dialogue_{interview_id}.png", dpi=150)
    plt.close(fig)


def plot_attention_entropy_by_class(
    entropies: list[float],
    labels: list[int],
    split_name: str,
    output_dir: Path,
) -> None:
    """Plot attention entropy distributions separated by class.

    Helps determine if the model attends differently to depressed vs.
    non-depressed dialogues.

    Args:
        entropies: Entropy values per dialogue.
        labels: True labels per dialogue.
        split_name: Split name for title and filename.
        output_dir: Directory to save the plot.
    """
    entropies = np.array(entropies)
    labels = np.array(labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        entropies[labels == 0], bins=20, alpha=0.6,
        label="Not Depressed (0)", color="steelblue", edgecolor="white",
    )
    ax.hist(
        entropies[labels == 1], bins=20, alpha=0.6,
        label="Depressed (1)", color="salmon", edgecolor="white",
    )
    ax.set_xlabel("Attention Entropy")
    ax.set_ylabel("Count")
    ax.set_title(f"Attention Entropy by Class — {split_name}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"attention_entropy_by_class_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_top_utterances_report(
    predictions: dict,
    split_name: str,
    output_dir: Path,
    top_k: int = 3,
    max_dialogues: int = 20,
) -> None:
    """Generate a text report of top-attended utterances per dialogue.

    Args:
        predictions: Predictions dict from evaluate() containing attention_weights,
                     utterance_texts, interview_id, true_label, probability.
        split_name: Split name.
        output_dir: Directory to save the report.
        top_k: Number of top utterances per dialogue.
        max_dialogues: Maximum dialogues to include.
    """
    lines = [f"# Top-{top_k} Attended Utterances — {split_name}\n"]

    n = min(max_dialogues, len(predictions["interview_id"]))
    for i in range(n):
        iv_id = predictions["interview_id"][i]
        label = predictions["true_label"][i]
        prob = predictions["probability"][i]
        weights = predictions["attention_weights"][i]
        texts = predictions["utterance_texts"][i]

        sorted_idx = np.argsort(weights)[::-1][:top_k]

        lines.append(f"\n## Interview {iv_id}")
        lines.append(f"- True: {label} | Pred prob: {prob:.4f} | Utterances: {len(weights)}")
        for rank, idx in enumerate(sorted_idx, 1):
            lines.append(f"  {rank}. [U{idx}] w={weights[idx]:.4f}: {texts[idx]}")

    report = "\n".join(lines)
    with open(output_dir / f"top_utterances_{split_name}.md", "w", encoding="utf-8") as f:
        f.write(report)
