import re
from collections import defaultdict
import numpy as np

# Parse v9d validation metrics from log
v9d_folds = defaultdict(list)
try:
    with open('results/ss_damil_r_cv_v9d_single_run_tuning/cv_run.log', 'r') as f:
        lines = f.readlines()
        
    run_best_f1 = defaultdict(float)
    run_to_fold = {}
    
    for line in lines:
        match = re.search(r'Run (\d+)', line)
        if match:
            run_seed = int(match.group(1))
            fold_idx = run_seed // 100
            run_to_fold[run_seed] = fold_idx
            
            f1_match = re.search(r'F1:([\d\.]+)', line)
            if f1_match:
                f1 = float(f1_match.group(1))
                run_best_f1[run_seed] = max(run_best_f1[run_seed], f1)

    for run, f1 in run_best_f1.items():
        v9d_folds[run_to_fold[run]].append(f1)

    print("=== v9d (Validation Metrics from Log) ===")
    for f_idx in sorted(v9d_folds.keys()):
        print(f"Fold {f_idx}: Val F1 (Best) = {np.mean(v9d_folds[f_idx]):.4f} (across {len(v9d_folds[f_idx])} seeds)")
except Exception as e:
    print(f"Error reading v9d log: {e}")

