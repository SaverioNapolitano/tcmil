"""DAMIL-CL: Clinical Linguistic Enhanced Dual Attention MIL.

Extends DAMIL-R with two orthogonal improvements:

1. **Dual-stream instance representation**
   - Stream A: frozen sentence embedding (768-dim) projected to `proj_dim`
     with cross-role attention as in DAMIL-R.
   - Stream B: 16 hand-crafted clinical linguistic features per utterance
     (disfluency, pronoun use, negation, affect, turn position, …)
     projected to `ling_proj_dim`.
   Both streams are concatenated into an `instance_dim`-dimensional
   per-utterance representation.

2. **ABMIL with gating + multi-view pooling**
   - ABMIL (Ilse et al., 2018): attention weights are gated by two parallel
     branches (tanh · sigmoid), making them more selective than a plain
     tanh attention scorer.
   - Attention-pooled representation is concatenated with the global mean
     pool, doubling the bag-level capacity at minimal parameter cost.

Architecture:
    Patient emb (P, 768) ──→ Proj ──→ CrossRoleAttn ──→ RoleAwareFusion ──→ (P, proj_dim)
                                                                               │
    Patient ling (P, 16) ──→ BN1d ──→ Proj ──→ ReLU ──────────────────────── │
                                                                               ↓
                                                                  concat → (P, instance_dim)
                                                                               │
                                                                  ABMIL-gate → attn_pool (instance_dim)
                                                                  mean_pool  → mean_pool (instance_dim)
                                                                               │
                                                                  concat → (2·instance_dim)
                                                                               │
                                                              Dropout → Linear → logit
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.damil_r import CrossRoleAttention, RoleAwareFusion


# ---------------------------------------------------------------------------
# Gated Attention Pooling (ABMIL)
# ---------------------------------------------------------------------------

class GatedAttentionPooling(nn.Module):
    """Attention-based MIL pooling with gating (Ilse et al., 2018, Eq. 9).

    Computes:
        V_k = tanh(W_V · h_k)              (P, att_hidden)
        U_k = sigmoid(W_U · h_k)           (P, att_hidden)
        e_k = w^T (V_k ⊙ U_k)              (P,)
        a   = softmax(e)                    (P,)
        z   = Σ_k a_k h_k                  (instance_dim,)

    The sigmoid gate allows the model to suppress irrelevant instances
    (when U_k ≈ 0) in a way a plain tanh cannot.

    Args:
        instance_dim: Dimensionality of each instance representation.
        att_hidden: Hidden size for the V and U branches.
    """

    def __init__(self, instance_dim: int, att_hidden: int = 64):
        super().__init__()
        self.W_V = nn.Linear(instance_dim, att_hidden)
        self.W_U = nn.Linear(instance_dim, att_hidden)
        self.w   = nn.Linear(att_hidden, 1, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Instance representations of shape (P, instance_dim).

        Returns:
            pooled: Attention-weighted bag representation (instance_dim,).
            weights: Per-instance attention weights (P,).
        """
        v = torch.tanh(self.W_V(x))        # (P, att_hidden)
        u = torch.sigmoid(self.W_U(x))     # (P, att_hidden)
        e = self.w(v * u)                  # (P, 1)
        a = torch.softmax(e, dim=0)        # (P, 1)
        pooled = (x * a).sum(dim=0)        # (instance_dim,)
        return pooled, a.squeeze(-1)       # (instance_dim,), (P,)


# ---------------------------------------------------------------------------
# DAMIL-CL Classifier
# ---------------------------------------------------------------------------

class DAMILCLClassifier(nn.Module):
    """Clinical Linguistic Enhanced Dual Attention MIL Classifier.

    Args:
        embedding_dim: Dimension of the frozen sentence embeddings (default: 768).
        ling_dim: Number of clinical linguistic features per utterance (default: 16).
        proj_dim: Embedding projection dimension (default: 64).
        ling_proj_dim: Linguistic feature projection dimension (default: 16).
        att_hidden: Hidden size for the ABMIL gating branches (default: 64).
        dropout_rate: Dropout applied before the final linear layer (default: 0.35).
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ling_dim: int = 16,
        proj_dim: int = 64,
        ling_proj_dim: int = 16,
        att_hidden: int = 64,
        dropout_rate: float = 0.35,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ling_dim = ling_dim
        instance_dim = proj_dim + ling_proj_dim

        # ── Stream A: sentence embedding ──────────────────────────────────
        self.emb_proj = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
            nn.ReLU(),
        )
        self.cross_attention = CrossRoleAttention(dim=proj_dim)
        self.fusion = RoleAwareFusion(dim=proj_dim)

        # ── Stream B: linguistic features ─────────────────────────────────
        # BatchNorm1d stabilizes the heterogeneous feature scales on small
        # datasets better than a learned scale per feature.
        self.ling_bn = nn.BatchNorm1d(ling_dim)
        self.ling_proj = nn.Sequential(
            nn.Linear(ling_dim, ling_proj_dim),
            nn.ReLU(),
        )

        # ── Aggregation ───────────────────────────────────────────────────
        self.attn_pool = GatedAttentionPooling(instance_dim, att_hidden)

        # Bag = [attention_pool ‖ mean_pool]  →  2 · instance_dim
        bag_dim = 2 * instance_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(bag_dim, 1),
        )

    # ------------------------------------------------------------------
    # Single-bag forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        patient_emb: torch.Tensor,
        patient_ling: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single interview bag.

        Args:
            patient_emb: (P, embedding_dim) patient utterance embeddings.
            patient_ling: (P, ling_dim) normalized linguistic features.
            interviewer_emb: (I, embedding_dim) interviewer embeddings.
            noise_std: Gaussian noise injected onto embeddings during training.

        Returns:
            logit: Scalar classification logit.
            cross_attn_weights: (P, I) cross-role attention matrix.
            turn_attn_weights: (P,) ABMIL attention weights.
        """
        # ── Embedding noise ───────────────────────────────────────────────
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std
            interviewer_emb = (
                interviewer_emb + torch.randn_like(interviewer_emb) * noise_std
            )

        # ── Stream A ──────────────────────────────────────────────────────
        patient_proj = self.emb_proj(patient_emb)         # (P, proj_dim)
        intv_proj    = self.emb_proj(interviewer_emb)     # (I, proj_dim)

        context, cross_attn = self.cross_attention(patient_proj, intv_proj)
        fused = self.fusion(patient_proj, context)        # (P, proj_dim)

        # ── Stream B ──────────────────────────────────────────────────────
        # BatchNorm1d requires batch dim; reshape (P, L) → process → reshape back
        ling_normed = self.ling_bn(patient_ling)          # (P, ling_dim)
        ling_out    = self.ling_proj(ling_normed)         # (P, ling_proj_dim)

        # ── Instance fusion ───────────────────────────────────────────────
        instance = torch.cat([fused, ling_out], dim=-1)  # (P, instance_dim)

        # ── Bag aggregation ───────────────────────────────────────────────
        attn_pooled, attn_weights = self.attn_pool(instance)   # (instance_dim,), (P,)
        mean_pooled = instance.mean(dim=0)                     # (instance_dim,)

        bag = torch.cat([attn_pooled, mean_pooled], dim=-1)   # (2·instance_dim,)

        logit = self.classifier(bag).squeeze(-1)

        return logit, cross_attn, attn_weights

    # ------------------------------------------------------------------
    # Batched forward (handles variable-length bags with padding)
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        patient_bags: torch.Tensor,
        patient_ling_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_sizes: list[int],
        interviewer_sizes: list[int],
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass for a padded batch of bags.

        Args:
            patient_bags: (B, max_P, embedding_dim) padded patient embeddings.
            patient_ling_bags: (B, max_P, ling_dim) padded linguistic features.
            interviewer_bags: (B, max_I, embedding_dim) padded interviewer embeddings.
            patient_sizes: Actual patient utterance counts per bag.
            interviewer_sizes: Actual interviewer utterance counts per bag.
            noise_std: Gaussian noise std for embeddings during training.

        Returns:
            logits: (B,) classification logits.
            cross_attn_list: List of (P_i, I_i) cross-attention tensors.
            turn_attn_list: List of (P_i,) ABMIL attention weight vectors.
        """
        logits = []
        cross_attn_list = []
        turn_attn_list  = []

        for i in range(patient_bags.size(0)):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]

            logit, cross_attn, turn_attn = self.forward(
                patient_emb=patient_bags[i, :p_n, :],
                patient_ling=patient_ling_bags[i, :p_n, :],
                interviewer_emb=interviewer_bags[i, :i_n, :],
                noise_std=noise_std,
            )
            logits.append(logit)
            cross_attn_list.append(cross_attn)
            turn_attn_list.append(turn_attn)

        return torch.stack(logits), cross_attn_list, turn_attn_list

    # ------------------------------------------------------------------
    # Backbone-only forward (for R-Drop / Mixup tricks)
    # ------------------------------------------------------------------

    def forward_backbone(
        self,
        patient_bags: torch.Tensor,
        patient_ling_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_sizes: list[int],
        interviewer_sizes: list[int],
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Return the bag-level representation (before classifier), batched.

        Used for R-Drop: the classifier head is applied twice on the same
        representation with different dropout masks.

        Returns:
            bag_repr: (B, 2·instance_dim)
        """
        bag_reprs = []
        for i in range(patient_bags.size(0)):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]

            patient_emb    = patient_bags[i, :p_n, :]
            patient_ling   = patient_ling_bags[i, :p_n, :]
            interviewer_emb = interviewer_bags[i, :i_n, :]

            if self.training and noise_std > 0:
                patient_emb    = patient_emb    + torch.randn_like(patient_emb)    * noise_std
                interviewer_emb = interviewer_emb + torch.randn_like(interviewer_emb) * noise_std

            patient_proj = self.emb_proj(patient_emb)
            intv_proj    = self.emb_proj(interviewer_emb)
            context, _   = self.cross_attention(patient_proj, intv_proj)
            fused        = self.fusion(patient_proj, context)

            ling_normed  = self.ling_bn(patient_ling)
            ling_out     = self.ling_proj(ling_normed)

            instance = torch.cat([fused, ling_out], dim=-1)

            attn_pooled, _ = self.attn_pool(instance)
            mean_pooled    = instance.mean(dim=0)
            bag = torch.cat([attn_pooled, mean_pooled], dim=-1)
            bag_reprs.append(bag)

        return torch.stack(bag_reprs)   # (B, 2·instance_dim)
