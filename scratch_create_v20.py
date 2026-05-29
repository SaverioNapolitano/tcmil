import re

with open("training/cv_ss_damil_r_v9d.py", "r") as f:
    text = f.read()

# 1. Update docstring
text = text.replace('"""SS-DAMIL-R v9d: Manifold Mixup (ablation).', '"""SS-DAMIL-R v20: The Dual-Path Distillation Pipeline.')
text = text.replace('keeps all v9c improvements', 'Keeps all v9d improvements and introduces Dual-Path pooling and BGE encoder.')

# 2. Update model import
text = text.replace('from models.ss_damil_r import SSDamilRClassifierV9', 'from models.ss_damil_r_v20 import SSDamilRClassifierV20')
text = text.replace('SSDamilRClassifierV9(', 'SSDamilRClassifierV20(')


# 3. Update train_epoch signature
text = text.replace('def train_epoch_v9d(', 'def train_epoch_v20(')
text = text.replace('train_epoch_v9d(', 'train_epoch_v20(')

# 4. Update arg parser defaults
text = text.replace('description="SS-DAMIL-R v9d: Manifold Mixup"', 'description="SS-DAMIL-R v20: Dual-Path Distillation Pipeline"')
text = text.replace('default="results/ss_damil_r_cv_v9d"', 'default="results/ss_damil_r_cv_v20"')
text = text.replace('default="sentence-transformers/all-mpnet-base-v2"', 'default="BAAI/bge-base-en-v1.5"')

# 5. Add prepend_roles argument and usage
prepend_arg = '    parser.add_argument("--prepend_roles", action="store_true", default=True)\n'
text = re.sub(r'(    parser.add_argument\("--encoder_name".*\n)', r'\1' + prepend_arg, text)

precompute_call = 'precompute_dual_role_embeddings(all_interviews, tokenizer, base_encoder, device, max_len=args.max_len)'
precompute_call_new = 'precompute_dual_role_embeddings(all_interviews, tokenizer, base_encoder, device, max_len=args.max_len, prepend_roles=args.prepend_roles)'
text = text.replace(precompute_call, precompute_call_new)

# 6. Update logger and report texts
text = text.replace('v9 architecture', 'v20 architecture')
text = text.replace('(v9d - Manifold Mixup)', '(v20 - Dual-Path)')
text = text.replace('SS-DAMIL-R v9d (Manifold Mixup)', 'SS-DAMIL-R v20 (Dual-Path Pooling)')

with open("training/cv_ss_damil_r_v20.py", "w") as f:
    f.write(text)

