"""Single-model calibration / decision figure (the pos_weight=1.0 mechanism).

The paper's central single-model claim: the macro-F1 gap is a DECISION/CALIBRATION
problem, not a ranking one. pos_weight=1.0 de-skews the recall-biased probabilities
so the fixed a-priori prevalence threshold lands near the oracle. Two complementary
panels on the seed-averaged test probabilities:

  (a) threshold -> macro-F1 curve for pos_weight=auto vs 1.0, with each model's
      prevalence-matched operating threshold (vertical) and oracle max (marker).
      Shows WHERE the decision lands relative to the achievable peak.
  (b) reliability diagram (5 quantile bins): mean predicted prob vs empirical
      positive frequency, diagonal = perfect calibration. Shows the recall-skew
      of the auto model and its correction at pos_weight=1.0.

(a) = decision geometry, (b) = calibration quality — complementary, not redundant.
"""

import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.training.train_tcmil_official import tune_threshold
from src.core.utils.metrics import compute_metrics

PREV = 0.28
AUTO = "results/single_model/lever_pos_weight/single_pw_auto_30seed/results.json"
PW1 = "results/single_model/headline_pw1_30seed/results.json"


def avg_probs(path):
    r = json.load(open(path))
    return np.mean([np.asarray(p) for p in r["test_prob_runs"]], axis=0), np.asarray(r["test_labels"])


def f1_curve(prob, y, ts):
    return np.array([compute_metrics(y, (prob >= t).astype(int), prob)["macro_f1"] for t in ts])


def reliability(prob, y, nbins=5):
    qs = np.quantile(prob, np.linspace(0, 1, nbins + 1))
    qs[0] -= 1e-9
    xs, ys = [], []
    for i in range(nbins):
        m = (prob > qs[i]) & (prob <= qs[i + 1])
        if m.sum() > 0:
            xs.append(prob[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def main():
    pa, y = avg_probs(AUTO)
    pp, _ = avg_probs(PW1)
    ts = np.linspace(0.02, 0.98, 97)
    fa, fp = f1_curve(pa, y, ts), f1_curve(pp, y, ts)
    ta = tune_threshold(None, pa, metric="prevalence", prevalence=PREV)
    tp = tune_threshold(None, pp, metric="prevalence", prevalence=PREV)
    fa_prev = compute_metrics(y, (pa >= ta).astype(int), pa)["macro_f1"]
    fp_prev = compute_metrics(y, (pp >= tp).astype(int), pp)["macro_f1"]
    print(f"auto : oracle {fa.max():.3f}@t={ts[fa.argmax()]:.2f}  prev-thr {fa_prev:.3f}@t={ta:.2f}")
    print(f"pw1.0: oracle {fp.max():.3f}@t={ts[fp.argmax()]:.2f}  prev-thr {fp_prev:.3f}@t={tp:.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=300)
    # (a) threshold -> F1
    ax[0].plot(ts, fa, color="#888888", lw=1.4, label="pos_weight=auto")
    ax[0].plot(ts, fp, color="#1f4e9c", lw=1.4, label="pos_weight=1.0")
    ax[0].axvline(ta, color="#888888", ls=":", lw=1)
    ax[0].axvline(tp, color="#1f4e9c", ls=":", lw=1)
    ax[0].plot(ts[fa.argmax()], fa.max(), "o", color="#888888", ms=4)
    ax[0].plot(ts[fp.argmax()], fp.max(), "o", color="#1f4e9c", ms=4)
    ax[0].scatter([ta, tp], [fa_prev, fp_prev], marker="x", s=40,
                  c=["#888888", "#1f4e9c"], zorder=5)
    ax[0].set_xlabel("decision threshold", fontsize=8)
    ax[0].set_ylabel("macro-F1", fontsize=8)
    ax[0].set_title("(a) threshold $\\to$ macro-F1", fontsize=9)
    ax[0].tick_params(labelsize=7)
    ax[0].text(0.5, 0.16, "$\\circ$ oracle  $\\times$ prevalence-threshold",
               transform=ax[0].transAxes, ha="center", fontsize=6, color="#444")
    ax[0].legend(fontsize=6.5, loc="lower center", frameon=False)
    # (b) reliability
    xa, ya = reliability(pa, y); xp, yp = reliability(pp, y)
    ax[1].plot([0, 1], [0, 1], ls="--", lw=0.8, color="#b22222")
    ax[1].plot(xa, ya, "o-", color="#888888", ms=4, lw=1.2, label="auto")
    ax[1].plot(xp, yp, "^-", color="#1f4e9c", ms=4, lw=1.2, label="1.0")
    ax[1].set_xlabel("mean predicted prob.", fontsize=8)
    ax[1].set_ylabel("empirical positive freq.", fontsize=8)
    ax[1].set_title("(b) reliability (5 quantile bins)", fontsize=9)
    ax[1].tick_params(labelsize=7)
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.tight_layout(pad=0.5)
    out = "paper/fig_calibration"
    fig.savefig(out + ".png", bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
