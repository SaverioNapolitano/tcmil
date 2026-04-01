"""DAMIL-R cross-attention diagnostics and visualization.

Provides functions for:
- Cross-attention heatmaps (patient turns × interviewer turns)
- Turn-level attention histograms
- Cross-attention entropy diagnostics
- Top cross-attended utterance reports
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")


def plot_cross_attention_heatmap(
    dialogue_id: int,
    attn_matrix: np.ndarray,
    true_label: int,
    predicted_prob: float,
    output_dir: Path,
    patient_texts: list[str] | None = None,
    interviewer_texts: list[str] | None = None,
    max_display: int = 30,
) -> None:
    """Plot a heatmap of the cross-attention matrix for a single dialogue.

    Shows how each patient turn attends to each interviewer turn.

    Args:
        dialogue_id: Participant ID.
        attn_matrix: Cross-attention matrix of shape (P, I).
        true_label: Ground truth label (0 or 1).
        predicted_prob: Model's predicted probability.
        output_dir: Directory to save the plot.
        patient_texts: Optional participant utterance strings.
        interviewer_texts: Optional interviewer utterance strings.
        max_display: Maximum number of turns to display on each axis.
    """
    P, I = attn_matrix.shape

    # Truncate for readability if needed
    p_display = min(P, max_display)
    i_display = min(I, max_display)
    matrix_display = attn_matrix[:p_display, :i_display]

    fig, ax = plt.subplots(figsize=(max(8, i_display * 0.4), max(6, p_display * 0.3)))

    sns.heatmap(
        matrix_display,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Attention Weight"},
        linewidths=0.3,
        linecolor="white",
    )

    # Y-axis: patient turn labels
    if patient_texts is not None:
        ylabels = [
            f"P{i}: {t[:30]}..." if len(t) > 30 else f"P{i}: {t}"
            for i, t in enumerate(patient_texts[:p_display])
        ]
    else:
        ylabels = [f"P{i}" for i in range(p_display)]
    ax.set_yticklabels(ylabels, fontsize=7, rotation=0)

    # X-axis: interviewer turn labels
    if interviewer_texts is not None:
        xlabels = [
            f"I{i}: {t[:20]}..." if len(t) > 20 else f"I{i}: {t}"
            for i, t in enumerate(interviewer_texts[:i_display])
        ]
    else:
        xlabels = [f"I{i}" for i in range(i_display)]
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")

    label_str = "Depressed" if true_label == 1 else "Not Depressed"
    ax.set_title(
        f"DAMIL-R Cross-Attention — Interview {dialogue_id}\n"
        f"True: {label_str} | Pred Prob: {predicted_prob:.3f} | "
        f"Patient turns: {P}, Interviewer turns: {I}",
        fontsize=10,
    )
    ax.set_xlabel("Interviewer Turns")
    ax.set_ylabel("Patient Turns")

    fig.tight_layout()
    fig.savefig(output_dir / f"cross_attention_heatmap_{dialogue_id}.png", dpi=150)
    plt.close(fig)


def plot_turn_attention_histogram(
    all_weights: list[np.ndarray],
    split_name: str,
    output_dir: Path,
) -> None:
    """Plot histogram of turn-level attention weights across dialogues.

    Shows the distribution of how attention mass is allocated across
    patient turns after fusion.

    Args:
        all_weights: List of turn attention weight arrays (one per dialogue).
        split_name: Split name for title and filename.
        output_dir: Directory to save the plot.
    """
    if not all_weights:
        return

    # In multi-head pooling, all_weights[i] is (num_heads, P_i)
    # We flatten everything to see the total mass distribution across all heads
    flat_weights = np.concatenate([w.ravel() for w in all_weights])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw weight distribution
    axes[0].hist(flat_weights, bins=50, color="teal", alpha=0.7, edgecolor="white")
    axes[0].set_xlabel("Turn Attention Weight")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Turn Attention Weight Distribution — {split_name}")
    axes[0].axvline(
        x=np.mean(flat_weights), color="red", linestyle="--",
        label=f"mean={np.mean(flat_weights):.4f}",
    )
    axes[0].legend()

    # Max weight per dialogue (concentration measure)
    max_weights = [w.max() for w in all_weights]
    axes[1].hist(max_weights, bins=30, color="coral", alpha=0.7, edgecolor="white")
    axes[1].set_xlabel("Max Turn Attention Weight per Dialogue")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Turn Attention Concentration — {split_name}")
    axes[1].axvline(
        x=np.mean(max_weights), color="red", linestyle="--",
        label=f"mean={np.mean(max_weights):.4f}",
    )
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"turn_attention_histogram_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_cross_attention_entropy_histogram(
    entropies: list[float],
    split_name: str,
    output_dir: Path,
) -> None:
    """Plot histogram of mean cross-attention entropy across dialogues.

    Low entropy indicates attention collapse (focusing on one interviewer turn).
    High entropy indicates diffuse attention.

    Args:
        entropies: Mean cross-attention entropy per dialogue.
        split_name: Split name for title and filename.
        output_dir: Directory to save the plot.
    """
    if not entropies:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropies, bins=20, color="mediumpurple", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Mean Cross-Attention Entropy")
    ax.set_ylabel("Count")
    ax.set_title(f"Cross-Attention Entropy Distribution — {split_name}")
    ax.axvline(
        x=np.mean(entropies), color="red", linestyle="--",
        label=f"mean={np.mean(entropies):.4f}",
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"cross_attention_entropy_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_cross_attention_entropy_by_class(
    entropies: list[float],
    labels: list[int],
    split_name: str,
    output_dir: Path,
) -> None:
    """Plot cross-attention entropy distributions separated by class.

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
    ax.set_xlabel("Mean Cross-Attention Entropy")
    ax.set_ylabel("Count")
    ax.set_title(f"Cross-Attention Entropy by Class — {split_name}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"cross_attention_entropy_by_class_{split_name}.png", dpi=150)
    plt.close(fig)


def generate_cross_attention_report(
    predictions: dict,
    split_name: str,
    output_dir: Path,
    top_k_patients: int = 5,
    top_k_interviewers: int = 3,
    max_dialogues: int = 20,
) -> None:
    """Generate a text report of top cross-attended interviewer utterances per dialogue.

    Args:
        predictions: Predictions dict from evaluate() with cross-attention data.
        split_name: Split name.
        output_dir: Directory to save the report.
        top_k_patients: Number of top patient turns to show.
        top_k_interviewers: Number of top interviewer turns per patient turn.
        max_dialogues: Maximum dialogues to include.
    """
    lines = [f"# DAMIL-R Cross-Attention Report — {split_name}\n"]

    n = min(max_dialogues, len(predictions["interview_id"]))
    for i in range(n):
        iv_id = predictions["interview_id"][i]
        label = predictions["true_label"][i]
        prob = predictions["probability"][i]
        cross_attn = predictions["cross_attention_weights"][i]  # (P, I)
        turn_attn = predictions["turn_attention_weights"][i]    # (P,)
        p_texts = predictions["utterance_texts"][i] if predictions["utterance_texts"] else None
        i_texts = predictions["interviewer_utterance_texts"][i] if predictions["interviewer_utterance_texts"] else None

        label_str = "Depressed" if label == 1 else "Not Depressed"
        lines.append(f"\n## Interview {iv_id} ({label_str}, Prob: {prob:.4f})")
        lines.append(f"- Patient turns: {len(turn_attn)}")
        lines.append(f"- Mean cross-attn entropy: {predictions['cross_attention_entropy'][i]:.4f}")
        lines.append(f"- Turn attn entropy: {predictions['turn_attention_entropy'][i]:.4f}")

        # If multi-head, average across heads for the summary report
        if turn_attn.ndim > 1:
            # turn_attn is (num_heads, P)
            # mean_turn_attn is (P,)
            mean_turn_attn = turn_attn.mean(axis=0)
        else:
            mean_turn_attn = turn_attn

        # Top patient turns by turn-level attention (using mean across heads)
        top_p = np.argsort(mean_turn_attn)[::-1][:top_k_patients]
        for rank, p_idx in enumerate(top_p, 1):
            p_text = p_texts[p_idx] if p_texts and p_idx < len(p_texts) else f"[turn {p_idx}]"
            lines.append(f"\n### P{p_idx} (mean_turn_attn={mean_turn_attn[p_idx]:.4f})")
            lines.append(f"> {p_text}")

            # Top interviewer turns this patient turn attends to
            cross_row = cross_attn[p_idx]
            top_i = np.argsort(cross_row)[::-1][:top_k_interviewers]
            for i_idx in top_i:
                i_text = i_texts[i_idx] if i_texts and i_idx < len(i_texts) else f"[turn {i_idx}]"
                lines.append(f"  - I{i_idx} (ca={cross_row[i_idx]:.4f}): {i_text}")

    report = "\n".join(lines)
    with open(output_dir / f"cross_attention_report_{split_name}.md", "w", encoding="utf-8") as f:
        f.write(report)
