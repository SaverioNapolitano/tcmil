"""Faithful per-chunk explanation for the BEST (plain attention-MIL) model.

Plain attention is near-uniform (entropy ratio ~1.0), so an attention heatmap is
uninformative and we do NOT plot it. But the plain head pools linearly, so its
leave-one-out (occlusion) Delta-prob is BOTH faithful (rho +0.99 vs attention)
AND informative (the LOO varies across chunks). This is the genuine single-chunk
attribution we can show for the plain model and not for the GRU one.

Output: one example interview's per-chunk occlusion saliency
(paper/fig_occlusion_saliency.{png,pdf}) + a top-k occluded-chunks text table
(paper/occlusion_topk.md) for the paper.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_example(per_iv):
    """A correctly-classified depressed interview with clear occlusion spread."""
    best, score = None, -1
    for iv in per_iv:
        pc = iv.get("per_chunk") or []
        if not pc or iv["label"] != 1 or iv["prob"] < 0.5:
            continue
        o = np.array([c["occ_delta"] for c in pc])
        spread = o.max() - o.min()
        if spread > score:
            best, score = iv, spread
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/interpretability/faithfulness_plain/interpret_test.json")
    ap.add_argument("--out", default="paper/fig_occlusion_saliency")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    data = json.load(open(args.json))
    iv = pick_example(data["per_interview"])
    if iv is None:
        print("no suitable example (need label=1, prob>0.5, per_chunk with text)")
        return
    pc = iv["per_chunk"]
    if "text" not in pc[0]:
        print("per_chunk has no text — rerun interpret_tcmil.py with the text dump")
        return
    occ = np.array([c["occ_delta"] for c in pc])
    n = len(occ)
    print(f"example interview {iv['interview_id']} label={iv['label']} prob={iv['prob']:.3f} "
          f"n_chunks={n} occ[{occ.min():.3f},{occ.max():.3f}]")

    # --- saliency strip: occlusion Delta-prob per chunk ---
    fig, ax = plt.subplots(figsize=(6.6, 1.5), dpi=300)
    vmax = np.abs(occ).max()
    ax.imshow(occ[None, :], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              extent=[0, n, 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("dialogue chunk index", fontsize=8)
    ax.set_title(f"Per-chunk occlusion saliency (plain MIL, interview "
                 f"{iv['interview_id']}, p={iv['prob']:.2f})", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"leave-one-out $\Delta$prob", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out + ".png", bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print(f"saved {args.out}.png / .pdf")

    # --- top-k occluded chunks horizontal bar plot (with text snippets) ---
    order_b = np.argsort(occ)[-args.topk:]  # ascending so largest on top
    figb, axb = plt.subplots(figsize=(6.6, 0.55 * args.topk + 0.7), dpi=300)
    vals = occ[order_b]
    labels = []
    for j in order_b:
        t = pc[j]["text"].replace("\n", " ")
        t = " ".join(t.split())
        # strip leading 'Participant:'/'Interviewer:' tags for compactness
        labels.append(f"[{j}] " + (t[:64] + "…" if len(t) > 64 else t))
    axb.barh(range(len(vals)), vals, color="#b22222")
    axb.set_yticks(range(len(vals)))
    axb.set_yticklabels(labels, fontsize=6)
    axb.set_xlabel(r"leave-one-out $\Delta$prob (chunk importance)", fontsize=8)
    axb.set_title("Most depression-supporting chunks (plain MIL, "
                  f"interview {iv['interview_id']}, p={iv['prob']:.2f})", fontsize=8)
    axb.tick_params(axis="x", labelsize=7)
    figb.tight_layout(pad=0.3)
    figb.savefig(args.out + "_topbars.png", bbox_inches="tight")
    figb.savefig(args.out + "_topbars.pdf", bbox_inches="tight")
    print(f"saved {args.out}_topbars.png / .pdf")

    # --- top-k occluded chunks text table ---
    order = np.argsort(-occ)[:args.topk]
    lines = [f"# Top-{args.topk} occluded chunks (plain MIL, faithful attribution)",
             f"Interview {iv['interview_id']}, label={iv['label']}, p={iv['prob']:.3f}\n",
             "| rank | chunk idx | LOO Δprob | text |", "| --- | --- | --- | --- |"]
    for r, j in enumerate(order, 1):
        txt = pc[j]["text"].replace("\n", " ").replace("|", "/")[:160]
        lines.append(f"| {r} | {j} | {occ[j]:+.3f} | {txt} |")
    tbl = Path(args.out + "_topk.md")
    tbl.write_text("\n".join(lines) + "\n")
    print(f"saved {tbl}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
