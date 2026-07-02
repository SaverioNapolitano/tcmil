"""Fig. 3 (real): pooled per-chunk attention vs leave-one-out occlusion delta,
contrasting the two TC-MIL pooling architectures.

The earlier fig:faith was a per-interview proxy (x=stability, y=per-interview
occlusion rho). interpret_tcmil.py now dumps each interview's per_chunk
[{attn, occ_delta}] pairs, so we draw the genuine per-chunk scatter: one point
per dialogue chunk, x = attention weight, y = leave-one-out Delta-prob.

Two panels (both pos_weight=1.0):
  (a) plain MIL  (temporal=none) -- the headline single model. Pooling is linear
      in instance embeddings, so LOO-delta tracks attention almost exactly:
      attention is faithful but near-uniform (trivial).
  (b) GRU MIL    (temporal=gru)  -- the temporal method variant. Recurrence
      mixes chunks, so removing one perturbs the others and the per-chunk
      attention<->occlusion link breaks (Jain & Wallace).

Faithfulness is thus a property of the pooling, not of attention per se.
Outputs PNG + PDF in paper/ and prints pooled stats for the caption.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pct_rank(v):
    """Within-interview percentile rank in [0,1]."""
    return (rankdata(v) - 1) / max(1, len(v) - 1)


def load_pairs(path):
    """Pool per-chunk pairs, rank-normalised WITHIN each interview.

    Faithfulness is a within-bag property: does the more-attended chunk also
    have the larger leave-one-out drop, *inside the same interview*. Pooling the
    raw weights across interviews triggers Simpson's paradox (the GRU model's
    within-bag rho is -0.28 but its raw-pooled rho is +0.51, because attention
    magnitude also separates interviews). Ranking within each interview removes
    the between-interview axis, so the pooled cloud's slope equals the metric.
    """
    data = json.load(open(path))
    attn, occ, lab, iv_rhos = [], [], [], []
    for iv in data["per_interview"]:
        pc = iv.get("per_chunk") or []
        if len(pc) < 4:
            continue
        a = np.array([c["attn"] for c in pc])
        o = np.array([c["occ_delta"] for c in pc])
        attn.append(_pct_rank(a)); occ.append(_pct_rank(o))
        lab.append(np.full(len(a), iv["label"]))
        r = spearmanr(a, o).correlation
        if r == r:
            iv_rhos.append(r)
    return (np.concatenate(attn), np.concatenate(occ),
            np.concatenate(lab), np.array(iv_rhos), len(data["per_interview"]))


def panel(ax, attn, occ, lab, iv_rhos, n_iv, title):
    rho_pool, p_pool = spearmanr(attn, occ)
    print(f"[{title}] n={len(attn)} chunks / {n_iv} interviews | "
          f"within-iv-rank pooled rho={rho_pool:.3f} (p={p_pool:.2g}) | "
          f"per-iv rho={iv_rhos.mean():.3f}+/-{iv_rhos.std():.3f}")
    for lv, col, mk in [(0, "#888888", "o"), (1, "#1f4e9c", "^")]:
        m = lab == lv
        ax.scatter(attn[m], occ[m], s=7, c=col, marker=mk, alpha=0.40,
                   linewidths=0)
    z = np.polyfit(attn, occ, 1)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, np.polyval(z, xs), lw=1.1, color="black")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#b22222")  # faithful diagonal
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("within-interview attention rank", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(0.96, 0.05, rf"$\rho={rho_pool:.2f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    return rho_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", default="results/interpretability/faithfulness_plain/interpret_test.json")
    ap.add_argument("--gru", default="results/interpretability/faithfulness_gru/interpret_test.json")
    ap.add_argument("--out", default="paper/fig3_faithfulness")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), dpi=300, sharey=True)
    panel(axes[0], *load_pairs(args.plain),
          title="(a) plain MIL (faithful, near-uniform)")
    panel(axes[1], *load_pairs(args.gru),
          title="(b) GRU MIL (not per-chunk faithful)")
    axes[0].set_ylabel("within-interview LOO-$\\Delta$prob rank", fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color="#888888", label="non-depressed"),
               Line2D([], [], marker="^", ls="", color="#1f4e9c", label="depressed")]
    axes[1].legend(handles=handles, fontsize=6, loc="upper right", frameon=False)
    fig.tight_layout(pad=0.4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out + ".png", bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print(f"saved {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
