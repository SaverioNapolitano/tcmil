"""DAMIL-X: Dialogue-aware Multi-View MIL for Depression Detection.

Extends DAMIL-R / DAMIL-CL with:
  1. Q-A dialogue pair instances (question + answer fused before encoding).
  2. Dual-stream instance representation (sentence emb + linguistic features).
  3. Multi-view MIL pooling: gated ABMIL + mean + max (3 complementary views).
  4. PHQ-8 sum regression auxiliary head on the shared bag representation.

All stages keep the MIL aggregation explicit (per-interview bag of Q-A pairs).

Architecture (single interview bag)::

    Patient Q-A emb (P, 768) ── emb_proj ──→ (P, d)
                                                │
                                                ↓
    Interviewer emb (I, 768) ── emb_proj ──→ (I, d)
                                                │
                    ┌──── L2-cos CrossRoleAttn ──┘
                    │
    (P, d)  context (P, d) ── RoleAwareFusion ──→ fused (P, d)
                                                │
    Patient ling (P, 16) ── BN → proj ──→ ling  (P, d_l)
                                                │
                    concat ───────────────────→ instance (P, d + d_l)
                                                │
                    Gated-ABMIL  → attn_pool   (d + d_l)
                    mean()       → mean_pool   (d + d_l)
                    max()        → max_pool    (d + d_l)
                                                │
                    concat → bag (3·(d + d_l))
                                                │
                    head_main → depression logit
                    head_aux  → PHQ-8 sum regression
"""

import torch
import torch.nn as nn

from models.damil_r import CrossRoleAttention, RoleAwareFusion


class GatedABMIL(nn.Module):
    """Gated Attention MIL pooling (Ilse et al., 2018, Eq. 9).

    scores = w^T (tanh(W_V h) ⊙ sigmoid(W_U h))
    weights = softmax(scores, dim=0)
    pooled = sum(weights * h)
    """

    def __init__(self, instance_dim: int, hidden: int = 64):
        super().__init__()
        self.W_V = nn.Linear(instance_dim, hidden)
        self.W_U = nn.Linear(instance_dim, hidden)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.tanh(self.W_V(x))
        u = torch.sigmoid(self.W_U(x))
        e = self.w(v * u)
        a = torch.softmax(e, dim=0)
        pooled = (x * a).sum(dim=0)
        return pooled, a.squeeze(-1)


class DAMILXClassifier(nn.Module):
    """Dialogue-aware multi-view MIL classifier.

    Args:
        embedding_dim: Frozen sentence-embedding dim (768 for mpnet).
        ling_dim: Number of hand-crafted linguistic features (16).
        proj_dim: Projection dim for sentence embeddings.
        ling_proj_dim: Projection dim for linguistic features.
        att_hidden: Hidden size for the gated-ABMIL scorer.
        dropout_rate: Dropout applied before both heads.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ling_dim: int = 16,
        proj_dim: int = 80,
        ling_proj_dim: int = 16,
        att_hidden: int = 64,
        dropout_rate: float = 0.35,
    ):
        super().__init__()
        self.instance_dim = proj_dim + ling_proj_dim
        bag_dim = 3 * self.instance_dim

        self.emb_proj = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
            nn.ReLU(),
        )
        self.cross_attention = CrossRoleAttention(dim=proj_dim)
        self.fusion = RoleAwareFusion(dim=proj_dim)

        self.ling_bn = nn.BatchNorm1d(ling_dim)
        self.ling_proj = nn.Sequential(
            nn.Linear(ling_dim, ling_proj_dim),
            nn.ReLU(),
        )

        self.pool_attn = GatedABMIL(self.instance_dim, hidden=att_hidden)

        self.dropout_main = nn.Dropout(dropout_rate)
        self.head_main = nn.Linear(bag_dim, 1)

        self.dropout_aux = nn.Dropout(dropout_rate)
        self.head_aux = nn.Linear(bag_dim, 1)

    def _bag_forward(
        self,
        patient_emb: torch.Tensor,
        patient_ling: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std
            interviewer_emb = interviewer_emb + torch.randn_like(interviewer_emb) * noise_std

        p_proj = self.emb_proj(patient_emb)
        i_proj = self.emb_proj(interviewer_emb)

        context, _ = self.cross_attention(p_proj, i_proj)
        fused = self.fusion(p_proj, context)

        if patient_ling.size(0) > 1:
            ling_n = self.ling_bn(patient_ling)
        else:
            ling_n = patient_ling
        ling_out = self.ling_proj(ling_n)

        instance = torch.cat([fused, ling_out], dim=-1)

        attn_pool, _ = self.pool_attn(instance)
        mean_pool = instance.mean(dim=0)
        max_pool = instance.max(dim=0).values

        return torch.cat([attn_pool, mean_pool, max_pool], dim=-1)

    def forward_backbone(
        self,
        patient_bags: torch.Tensor,
        patient_ling_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_sizes: list[int],
        interviewer_sizes: list[int],
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Return stacked bag representations of shape (B, 3·instance_dim)."""
        bag_reprs = []
        B = patient_bags.size(0)
        for i in range(B):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]
            bag = self._bag_forward(
                patient_bags[i, :p_n, :],
                patient_ling_bags[i, :p_n, :],
                interviewer_bags[i, :i_n, :],
                noise_std=noise_std,
            )
            bag_reprs.append(bag)
        return torch.stack(bag_reprs)

    def forward_heads(self, bag: torch.Tensor) -> dict:
        """Apply main and auxiliary heads on a bag representation.

        Each call samples fresh dropout masks (needed for R-Drop).
        """
        logit = self.head_main(self.dropout_main(bag)).squeeze(-1)
        aux = self.head_aux(self.dropout_aux(bag)).squeeze(-1)
        return {"logits": logit, "aux": aux}

    def forward(
        self,
        patient_bags: torch.Tensor,
        patient_ling_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_sizes: list[int],
        interviewer_sizes: list[int],
        noise_std: float = 0.0,
    ) -> dict:
        bag = self.forward_backbone(
            patient_bags, patient_ling_bags, interviewer_bags,
            patient_sizes, interviewer_sizes, noise_std=noise_std,
        )
        out = self.forward_heads(bag)
        out["bag"] = bag
        return out
