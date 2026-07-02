"""Build the TC-MIL design-ablation table from full189 + pw1.0 dev runs.

Reads results/design_ablations/grid/<config>/results.json (per_seed_dev has macro_f1,
roc_auc, f1). Emits a markdown summary and a LaTeX tabular for the paper.
Baseline = enc_bgelarge (== tmp_gru == ws_w4s2), printed once and referenced.
"""

import json
import os
import numpy as np

ROOT = "results/design_ablations/grid"

GROUPS = [
    ("Encoder (GRU, w4 s2)", [
        ("enc_bgelarge", "bge-large *(base)*"),
        ("enc_mxbai", "mxbai-large"),
        ("enc_uae", "UAE-large"),
        ("enc_gtelarge", "gte-large"),
        ("enc_e5large", "e5-large"),
        ("enc_bgebase", "bge-base"),
        ("enc_mpnet", "all-mpnet"),
    ]),
    ("Temporal head (bge-large)", [
        ("tmp_none", "none (plain)"),
        ("enc_bgelarge", "GRU *(base)*"),
        ("tmp_gru2", "GRU 2-layer"),
        ("tmp_trans", "Transformer"),
    ]),
    ("Window / stride (bge-large GRU)", [
        ("ws_w2s1", "w2 s1"),
        ("enc_bgelarge", "w4 s2 *(base)*"),
        ("ws_w6s3", "w6 s3"),
        ("ws_w4s1", "w4 s1"),
    ]),
    ("Regularisation (bge-large GRU, w4 s2)", [
        ("reg_drop03", "dropout 0.3"),
        ("enc_bgelarge", "dropout 0.4 *(base)*"),
        ("reg_drop05", "dropout 0.5"),
        ("reg_proj192", "proj 192"),
        ("reg_aux015", "aux 0.15"),
        ("enc_bgelarge", "aux 0.30 *(base)*"),
        ("reg_aux05", "aux 0.50"),
        ("reg_noaux", "aux 0.00"),
    ]),
]


def stats(name):
    f = os.path.join(ROOT, name, "results.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f))
    ps = r.get("per_seed_dev") or []
    def m(k):
        v = [s[k] for s in ps if s.get(k) is not None]
        return (np.mean(v), np.std(v)) if v else (None, None)
    de = r.get("dev_ensemble", {})
    return {"auc": m("roc_auc"), "maf1": m("macro_f1"), "f1": m("f1"),
            "ens_auc": de.get("roc_auc"), "ens_maf1": de.get("macro_f1")}


def fmt(t):
    return f"{t[0]:.3f}±{t[1]:.3f}" if t and t[0] is not None else "--"


def main():
    print("## TC-MIL design ablations (full189 dev, 5-seed, pos_weight=1.0)\n")
    for gname, configs in GROUPS:
        print(f"### {gname}\n")
        print("| config | dev AUC | dev macro-F1 | dev pos-F1 | ens AUC |")
        print("| --- | --- | --- | --- | --- |")
        for cfg, label in configs:
            s = stats(cfg)
            if s is None:
                print(f"| {label} | _pending_ | | | |"); continue
            print(f"| {label} | {fmt(s['auc'])} | {fmt(s['maf1'])} | "
                  f"{fmt(s['f1'])} | {fmt((s['ens_auc'],0)) if s['ens_auc'] else '--'} |")
        print()

    # LaTeX (encoder + temporal + window compact, macro-F1 + AUC)
    print("\n% --- LaTeX tabular (paste into paper) ---")
    print("\\begin{tabular}{@{}lcc@{}}\n\\toprule")
    print("\\textbf{Variant} & \\textbf{dev AUC} & \\textbf{dev macro-F1} \\\\")
    for gname, configs in GROUPS:
        print("\\midrule \\multicolumn{3}{@{}l}{\\emph{" + gname + "}} \\\\")
        seen = set()
        for cfg, label in configs:
            if cfg in seen:  # avoid printing shared baseline twice within group is fine
                pass
            s = stats(cfg)
            lab = label.replace("*(base)*", "(base)")
            if s is None:
                print(f"{lab} & -- & -- \\\\"); continue
            print(f"{lab} & {fmt(s['auc'])} & {fmt(s['maf1'])} \\\\")
    print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
