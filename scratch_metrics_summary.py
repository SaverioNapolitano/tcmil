import os
import glob
import re

results_dir = '/Users/saverionapolitano/Desktop/DAMIL-2/results'
pattern = os.path.join(results_dir, 'ss_damil_r_cv_v*', 'cv_report.txt')

print("Variant\t\t\tROC-AUC\t\tF1\t\tNote")
print("-" * 80)

metrics_data = []

for filepath in glob.glob(pattern):
    variant_name = os.path.basename(os.path.dirname(filepath))
    
    with open(filepath, 'r') as f:
        content = f.read()
        
    roc_auc_match = re.search(r'roc_auc\s+\|\s+([0-9\.]+)', content)
    f1_match = re.search(r'f1\s+\|\s+([0-9\.]+)', content)
    note_match = re.search(r'# SS-DAMIL-R .* \((.*?)\)', content)
    
    if roc_auc_match and f1_match:
        roc_auc = float(roc_auc_match.group(1))
        f1 = float(f1_match.group(1))
        note = note_match.group(1) if note_match else ""
        metrics_data.append((variant_name, roc_auc, f1, note))
        
metrics_data.sort(key=lambda x: x[1], reverse=True)

for name, roc_auc, f1, note in metrics_data:
    print(f"{name:<20}\t{roc_auc:.4f}\t\t{f1:.4f}\t\t{note}")
