"""SS-DAMIL-R v18: Lean MIL with Dual Regularization.

Ground-up redesign targeting the v9d baseline (ROC-AUC 0.8245, F1 0.6290).

Design philosophy: maximize signal extraction per parameter on a 189-subject
dataset. Every component must earn its place.

Architecture:
    Patient embeddings (P, 768) → Projector (768→64, ReLU) → (P, 64) ─┐
                                                                         ├─ CrossRoleAttention → Context (P, 64)
    Interviewer embeddings (I, 768) → Projector (shared) → (I, 64) ────┘
                                                                            │
    Patient (P, 64) + Context (P, 64) → RoleAwareFusion → LayerNorm → (P, 64)
                                                          │
                                                 AttentionPooling → (64,)
                                                          │
                                              Dropout(0.4) → Linear(64, 1) → logit

Key differences from v9d:
    - No symptom heads (removes ~1300 params & auxiliary gradient competition)
    - No multi-head pooling (single head is more stable on small data)
    - No gated symptom injection (eliminates 8→64 projection + gate)
    - Higher dropout (0.4) pre-classifier
    - Clean forward_backbone / forward_heads split for Manifold Mixup + R-Drop
    - ~30K trainable parameters (vs ~50K+ for v9d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.damil_r import CrossRoleAttention, RoleAwareFusion, AttentionPooling


class SSDamilRClassifierV18(nn.Module):
    """v18: Lean MIL classifier for depression detection.

    Combines the proven cross-role attention backbone with maximum simplicity
    in the classification head. Designed for Manifold Mixup + R-Drop training.

    Args:
        embedding_dim: Dimension of input sentence embeddings (768 for mpnet).
        proj_dim: Projection dimension for the shared low-dim space.
        dropout_rate: Dropout rate applied before classification.
        att_hidden_dim: Hidden dimension for the attention pooling scorer.
        temperature: Softmax temperature for attention weights.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 64,
        dropout_rate: float = 0.4,
        att_hidden_dim: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.working_dim = proj_dim if (proj_dim and proj_dim > 0) else embedding_dim

        # 1. Shared projection into low-dimensional space
        if proj_dim and proj_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.ReLU(),
            )
        else:
            self.projector = nn.Identity()

        # 2. Cross-role attention (parameter-free cosine attention)
        self.cross_attention = CrossRoleAttention(dim=self.working_dim)

        # 3. Gated residual fusion
        self.fusion = RoleAwareFusion(dim=self.working_dim)
        self.post_fusion_norm = nn.LayerNorm(self.working_dim)

        # 4. Single-head attention pooling
        self.pooling = AttentionPooling(
            input_dim=self.working_dim,
            hidden_dim=att_hidden_dim,
            temperature=temperature,
        )

        # 5. Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.working_dim, 1)

    def forward_backbone(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Run the backbone to produce a pooled bag representation.

        Args:
            patient_emb: Patient turn embeddings of shape (P, embedding_dim).
            interviewer_emb: Interviewer turn embeddings of shape (I, embedding_dim).
            noise_std: Gaussian noise std dev for training augmentation.

        Returns:
            pooled: Pooled bag representation of shape (D,).
        """
        # Optional embedding noise during training
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std
            interviewer_emb = interviewer_emb + torch.randn_like(interviewer_emb) * noise_std

        # Project both roles into shared space
        p_proj = self.projector(patient_emb)
        i_proj = self.projector(interviewer_emb)

        # Cross-role attention: patient queries interviewer
        context, _ = self.cross_attention(p_proj, i_proj)

        # Gated fusion with residual
        instance_features = self.post_fusion_norm(self.fusion(p_proj, context))

        # Attention-weighted pooling
        pooled, _ = self.pooling(instance_features)

        return pooled

    def forward_heads(self, pooled: torch.Tensor) -> dict:
        """Run the classification head on pooled representations.

        Supports both single (D,) and batched (B, D) inputs.
        Dropout is applied before classification — this means two calls
        during training produce different outputs (used for R-Drop).

        Args:
            pooled: Tensor of shape (D,) or (B, D).

        Returns:
            Dict with 'logit' (unbatched) or 'logits' (batched).
        """
        is_batched = pooled.dim() == 2

        h = self.dropout(pooled)
        logit = self.classifier(h).squeeze(-1)

        key = "logits" if is_batched else "logit"
        return {key: logit}

    def forward(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> dict:
        """Full forward pass for a single interview bag.

        Args:
            patient_emb: (P, embedding_dim).
            interviewer_emb: (I, embedding_dim).
            noise_std: Embedding noise for training.

        Returns:
            Dict with 'logit' and 'pooled_representation'.
        """
        pooled = self.forward_backbone(patient_emb, interviewer_emb, noise_std)
        head_output = self.forward_heads(pooled)
        return {
            "logit": head_output.get("logit", head_output.get("logits")),
            "pooled_representation": pooled,
        }

    def forward_batch(
        self,
        patient_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_sizes: list[int],
        interviewer_sizes: list[int],
        noise_std: float = 0.0,
    ) -> dict:
        """Forward pass for a padded batch of dual-role bags.

        Args:
            patient_bags: (batch_size, max_P, embedding_dim).
            interviewer_bags: (batch_size, max_I, embedding_dim).
            patient_sizes: Number of valid patient turns per bag.
            interviewer_sizes: Number of valid interviewer turns per bag.
            noise_std: Embedding noise for training.

        Returns:
            Dict with 'logits' (batch_size,) and 'pooled' (batch_size, D).
        """
        batch_size = patient_bags.size(0)
        logits = []
        pooled_list = []

        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]

            p_i = patient_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]

            res = self.forward(p_i, i_i, noise_std=noise_std)
            logits.append(res["logit"])
            pooled_list.append(res["pooled_representation"])

        return {
            "logits": torch.stack(logits),
            "pooled": torch.stack(pooled_list),
        }

    def forward_batch_split(
        self,
        patient_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_sizes: list[int],
        interviewer_sizes: list[int],
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Run backbone only for the batch — returns pooled representations.

        Used by Manifold Mixup training to get pooled repr before
        running classification heads separately.

        Args:
            patient_bags: (batch_size, max_P, embedding_dim).
            interviewer_bags: (batch_size, max_I, embedding_dim).
            patient_sizes: Number of valid patient turns per bag.
            interviewer_sizes: Number of valid interviewer turns per bag.
            noise_std: Embedding noise for training.

        Returns:
            pooled_batch: Tensor of shape (batch_size, D).
        """
        batch_size = patient_bags.size(0)
        pooled_list = []

        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]
            p_i = patient_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]

            pooled = self.forward_backbone(p_i, i_i, noise_std=noise_std)
            pooled_list.append(pooled)

        return torch.stack(pooled_list)
