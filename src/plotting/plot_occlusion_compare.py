"""Faithful per-chunk attribution for the plain attention-MIL head, as TEXT.

Replaces the chunk-index saliency strip with a text-labelled top-k bar figure:
two panels at full text width (double-column figure*) so the chunk text stays
readable. (a) full-dialogue model, (b) participant-only model (bias robustness).
The bars are the leave-one-out Delta-prob (faithful chunk importance for the
linear plain head); the y-labels are the chunk text.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_example(per_iv):
    best, score = None, -1
    for iv in per_iv:
        pc = iv.get("per_chunk") or []
        if not pc or iv["label"] != 1 or iv["prob"] < 0.5 or "text" not in pc[0]:
            continue
        o = np.array([c["occ_delta"] for c in pc])
        if o.max() - o.min() > score:
            best, score = iv, o.max() - o.min()
    return best


def clean(t, n=90):
    t = " ".join(t.replace("\n", " ").split())
    return t[:n] + "…" if len(t) > n else t


def panel(ax, iv, title, k):
    pc = iv["per_chunk"]
    occ = np.array([c["occ_delta"] for c in pc])
    order = np.argsort(occ)[-k:]               # ascending -> largest on top
    vals = occ[order]
    labels = [clean(pc[j]["text"]) for j in order]
    ax.barh(range(k), vals, color="#b22222")
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(r"leave-one-out $\Delta$prob (chunk importance)", fontsize=8)
    ax.set_title(f"{title} (interview {iv['interview_id']}, $p$={iv['prob']:.2f})",
                 fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.margins(y=0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogue", default="results/interpretability/faithfulness_plain/interpret_test.json")
    ap.add_argument("--ponly", default="results/interpretability/faithfulness_ponly/interpret_test.json")
    ap.add_argument("--out", default="paper/fig_occlusion_attribution")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    da = pick_example(json.load(open(args.dialogue))["per_interview"])
    po = pick_example(json.load(open(args.ponly))["per_interview"])
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), dpi=300)
    panel(axes[0], da, "(a) full-dialogue model", args.topk)
    panel(axes[1], po, "(b) participant-only model (bias robustness)", args.topk)
    fig.tight_layout(pad=0.5, h_pad=1.4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out + ".png", bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print(f"saved {args.out}.png / .pdf   dialogue={da['interview_id']} ponly={po['interview_id']}")


if __name__ == "__main__":
    main()
