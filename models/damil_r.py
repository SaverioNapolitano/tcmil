"""DAMIL-R: Role-Aware Dual Attention MIL model for binary interview-level classification.

This model operates on pre-computed utterance embeddings for **both** participant and
interviewer turns. It uses cross-role attention (patient queries, interviewer keys/values)
to produce role-aware contextualized patient representations, which are then pooled via
learned attention and classified.

Architecture:
    Patient embeddings   (P, d)  ──┐
                                   ├─ CrossRoleAttention ──→ Context (P, d)
    Interviewer embeddings (I, d) ─┘                            │
                                                                │
    Patient (P, d) ── concat ── Context (P, d) ──→ RoleAwareFusion ──→ Fused (P, d)
                                                        │
                                               AttentionPooling ──→ (d,)
                                                        │
                                                 Dropout + Linear ──→ logit
"""

import torch
import torch.nn as nn


class CrossRoleAttention(nn.Module):
    """Single-head cross-attention from patient turns to interviewer turns.

    Computes:
        Q = patient embeddings          (P, d)
        K = interviewer embeddings      (I, d)
        V = interviewer embeddings      (I, d)
        attention_scores = softmax(QK^T / sqrt(d))   (P, I)
        context = attention_scores @ V               (P, d)

    Args:
        embedding_dim: Dimensionality of input embeddings.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.scale = embedding_dim ** 0.5

    def forward(
        self,
        patient: torch.Tensor,
        interviewer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-role attention.

        Args:
            patient: Patient turn embeddings of shape (P, d).
            interviewer: Interviewer turn embeddings of shape (I, d).

        Returns:
            context: Contextualized patient representations of shape (P, d).
            attention_weights: Cross-attention matrix of shape (P, I).
        """
        # scores: (P, I)
        scores = torch.matmul(patient, interviewer.t()) / self.scale

        # attention_weights: (P, I)
        attention_weights = torch.softmax(scores, dim=-1)

        # context: (P, d)
        context = torch.matmul(attention_weights, interviewer)

        return context, attention_weights


class RoleAwareFusion(nn.Module):
    """Fuse patient embeddings with cross-role context via concatenation + projection.

    Computes:
        fused = Linear(concat(patient, context))

    Args:
        embedding_dim: Dimensionality of each input (patient and context are same dim).
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.projection = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(
        self, patient: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Fuse patient and context representations.

        Args:
            patient: Original patient embeddings of shape (P, d).
            context: Cross-attention context of shape (P, d).

        Returns:
            Fused representation of shape (P, d).
        """
        # (P, 2d) -> (P, d)
        concatenated = torch.cat([patient, context], dim=-1)
        return self.projection(concatenated)


class AttentionPooling(nn.Module):
    """Learned attention pooling over a set of instance embeddings.

    Computes:
        e_i = tanh(W @ x_i + b)
        a_i = softmax(v^T @ e_i / temperature)
        output = sum(a_i * x_i)

    Args:
        input_dim: Dimensionality of input embeddings.
        hidden_dim: Hidden dimension for the attention scorer.
        temperature: Softmax temperature (higher = smoother weights).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.W = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted pooling.

        Args:
            x: Instance embeddings of shape (num_instances, input_dim).
            mask: Optional boolean mask of shape (num_instances,).
                  True = valid instance, False = padding to ignore.

        Returns:
            pooled: Weighted sum embedding of shape (input_dim,).
            weights: Attention weights of shape (num_instances,).
        """
        # e: (num_instances, hidden_dim)
        e = torch.tanh(self.W(x))
        # scores: (num_instances, 1)
        scores = self.v(e)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # weights: (num_instances, 1)
        weights = torch.softmax(scores / self.temperature, dim=0)

        # pooled: (input_dim,)
        pooled = (weights * x).sum(dim=0)

        return pooled, weights.squeeze(-1)


class DAMILRClassifier(nn.Module):
    """Role-Aware Dual Attention MIL classifier for depression detection.

    Wires together: CrossRoleAttention → RoleAwareFusion → AttentionPooling → classifier.

    Args:
        embedding_dim: Dimension of input utterance embeddings (default: 768).
        proj_dim: Optional projection dimension. If set, project embeddings before
                  cross-attention. If None, operate on raw embeddings.
        att_hidden_dim: Hidden dimension for the turn-level attention scorer.
        dropout_rate: Dropout rate applied after attention pooling.
        temperature: Softmax temperature for turn-level attention weights.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int | None = None,
        att_hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Optional linear projection before cross-attention
        if proj_dim is not None and proj_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.ReLU(),
            )
            working_dim = proj_dim
        else:
            self.projector = nn.Identity()
            working_dim = embedding_dim

        self.cross_attention = CrossRoleAttention(embedding_dim=working_dim)
        self.fusion = RoleAwareFusion(embedding_dim=working_dim)

        self.attention_pooling = AttentionPooling(
            input_dim=working_dim,
            hidden_dim=att_hidden_dim,
            temperature=temperature,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(working_dim, 1)

    def forward(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single dialogue.

        Args:
            patient_emb: Patient turn embeddings of shape (P, embedding_dim).
            interviewer_emb: Interviewer turn embeddings of shape (I, embedding_dim).

        Returns:
            logit: Scalar logit for binary classification.
            cross_attention_weights: Cross-attention matrix of shape (P, I).
            turn_attention_weights: Turn-level attention weights of shape (P,).
        """
        # Project both roles
        patient = self.projector(patient_emb)
        interviewer = self.projector(interviewer_emb)

        # Cross-role attention: patient queries interviewer
        context, cross_attn_weights = self.cross_attention(patient, interviewer)

        # Fuse patient + context
        fused = self.fusion(patient, context)

        # Pool across patient turns
        pooled, turn_attn_weights = self.attention_pooling(fused)

        # Classify
        pooled = self.dropout(pooled)
        logit = self.classifier(pooled).squeeze(-1)

        return logit, cross_attn_weights, turn_attn_weights

    def forward_batch(
        self,
        patient_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_sizes: list[int],
        interviewer_sizes: list[int],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass for a padded batch of dual-role bags.

        Args:
            patient_bags: (batch_size, max_P, embedding_dim).
            interviewer_bags: (batch_size, max_I, embedding_dim).
            patient_sizes: Number of valid patient turns per bag.
            interviewer_sizes: Number of valid interviewer turns per bag.

        Returns:
            logits: Tensor of shape (batch_size,).
            cross_attention_list: List of (P_i, I_i) tensors.
            turn_attention_list: List of (P_i,) tensors.
        """
        batch_size = patient_bags.size(0)
        logits = []
        cross_attention_list = []
        turn_attention_list = []

        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]

            patient_i = patient_bags[i, :p_n, :]
            interviewer_i = interviewer_bags[i, :i_n, :]

            logit, cross_attn, turn_attn = self.forward(patient_i, interviewer_i)

            logits.append(logit)
            cross_attention_list.append(cross_attn)
            turn_attention_list.append(turn_attn)

        logits = torch.stack(logits)
        return logits, cross_attention_list, turn_attention_list
