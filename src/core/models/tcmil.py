"""TC-MIL: gated-attention MIL over dialogue-chunk embeddings.

Deliberately minimal (~120K params). Prior DAMIL-R iterations showed that on
~107 training subjects, every extra mechanism (mixup, focal, SWA, diversity
losses, multi-head pooling) added variance without moving the ceiling. The
MIL component here is the classic gated attention of Ilse & Welling (2018).
"""

import torch
import torch.nn as nn


class GatedAttentionPooling(nn.Module):
    """Gated attention MIL pooling: a = softmax(w^T (tanh(Vh) * sigm(Uh)))."""

    def __init__(self, dim: int, attn_dim: int = 64):
        super().__init__()
        self.V = nn.Linear(dim, attn_dim)
        self.U = nn.Linear(dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        """h: (B, N, d), mask: (B, N) with 1 for real instances."""
        scores = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)  # (B, d)
        return pooled, attn


class TCMIL(nn.Module):
    """Projector -> gated attention MIL -> classifier (+ symptom aux head).

    The symptom head predicts the 8 binarized PHQ-8 items (score >= 1) from
    the pooled representation. It is a pure regularizer: it shares the
    backbone but does not feed the main logit (the v8.2 lesson).
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 128,
        attn_dim: int = 64,
        dropout: float = 0.4,
        temporal: str = "none",
        gru_layers: int = 1,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = temporal
        if temporal == "gru":
            self.context = nn.GRU(
                proj_dim, proj_dim // 2, num_layers=gru_layers,
                batch_first=True, bidirectional=True,
                dropout=dropout if gru_layers > 1 else 0.0,
            )
            self.context_norm = nn.LayerNorm(proj_dim)
        elif temporal == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=proj_dim, nhead=4, dim_feedforward=proj_dim * 2,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.context = nn.TransformerEncoder(layer, num_layers=1)
        elif temporal != "none":
            raise ValueError(f"unknown temporal mode: {temporal}")
        self.pool = GatedAttentionPooling(proj_dim, attn_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(proj_dim, 1)
        self.symptom_head = nn.Linear(proj_dim, 8)

    def forward(self, bags: torch.Tensor, mask: torch.Tensor):
        """bags: (B, N, embedding_dim), mask: (B, N)."""
        h = self.projector(bags)
        if self.temporal == "gru":
            lengths = mask.sum(1).long().clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths, batch_first=True, enforce_sorted=False)
            ctx, _ = self.context(packed)
            ctx, _ = nn.utils.rnn.pad_packed_sequence(
                ctx, batch_first=True, total_length=h.size(1))
            h = self.context_norm(h + ctx)  # residual keeps the order-free path
        elif self.temporal == "transformer":
            h = self.context(h, src_key_padding_mask=(mask == 0))
        pooled, attn = self.pool(h, mask)
        pooled = self.norm(pooled)
        z = self.dropout(pooled)
        return {
            "logits": self.classifier(z).squeeze(-1),
            "symptom_logits": self.symptom_head(z),
            "attention": attn,
        }
