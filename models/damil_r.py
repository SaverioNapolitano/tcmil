"""DAMIL-R: Role-Aware Dual Attention MIL model for binary interview-level classification.

This model operates on pre-computed utterance embeddings for **both** participant and
interviewer turns. It uses cross-role attention (patient queries, interviewer keys/values)
to produce role-aware contextualized patient representations, which are then pooled via
learned attention and classified.

Design for small dataset (~140 interviews):
    - Shared projection maps both roles to a common low-dim space
    - Cross-attention is parameter-free (scaled dot-product in projected space)
    - Fusion uses a single linear layer with LayerNorm for training stability
    - All model complexity is concentrated in the shared projector

Architecture:
    Patient embeddings   (P, 768)  ──→ Projector ──→  (P, d)  ──┐
                                                                  ├─ CrossRoleAttention ──→ Context (P, d)
    Interviewer embeddings (I, 768)  ──→ Projector ──→  (I, d)  ─┘           │
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
    """Parameter-free scaled dot-product cross-attention from patient to interviewer.

    Since both roles are already projected into a shared embedding space by the
    upstream projector, the raw dot-product between projected patient and
    interviewer embeddings is meaningful. No additional Q/K/V projections are
    needed — this keeps the parameter count low, critical for small datasets.

    Computes:
        attention_scores = softmax(patient @ interviewer^T / sqrt(d))   (P, I)
        context = attention_scores @ interviewer                        (P, d)

    Args:
        dim: Dimensionality of the shared projection space.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5

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
    """Fuse patient embeddings with cross-role context.

    Uses concatenation + linear projection with LayerNorm for stability.
    A residual connection from the patient representation ensures the model
    can fall back to ignoring the interviewer context when it's not helpful.

        fused = LayerNorm(patient + Linear(concat(patient, context)))

    Args:
        dim: Dimensionality of both patient and context embeddings.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Linear(2 * dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

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
        concatenated = torch.cat([patient, context], dim=-1)
        projected = self.projection(concatenated)
        fused = self.layer_norm(patient + projected)

        return fused


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

    Wires together: Projector → CrossRoleAttention → RoleAwareFusion → AttentionPooling → classifier.

    Design principle: keep the cross-attention parameter-free and concentrate all
    learnable capacity in the shared projector and attention pooling. This is
    critical for small datasets where learned Q/K/V projections overfit rapidly.

    Args:
        embedding_dim: Dimension of input utterance embeddings (default: 768).
        proj_dim: Projection dimension for the shared embedding space.
                  If None or 0, operates on raw embeddings.
        att_hidden_dim: Hidden dimension for the turn-level attention scorer.
        dropout_rate: Dropout rate applied after projection and before classification.
        temperature: Softmax temperature for turn-level attention weights.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int | None = None,
        att_hidden_dim: int = 64,
        dropout_rate: float = 0.3,
        temperature: float = 1.0,
        # Legacy params accepted but ignored for backward compat
        attn_dim: int | None = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Shared projection into a low-dimensional common space
        if proj_dim is not None and proj_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.ReLU(),
            )
            working_dim = proj_dim
        else:
            self.projector = nn.Identity()
            working_dim = embedding_dim

        # Parameter-free cross-attention in the projected space
        self.cross_attention = CrossRoleAttention(dim=working_dim)

        # Fusion with residual connection + LayerNorm
        self.fusion = RoleAwareFusion(dim=working_dim)

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
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single dialogue.

        Args:
            patient_emb: Patient turn embeddings of shape (P, embedding_dim).
            interviewer_emb: Interviewer turn embeddings of shape (I, embedding_dim).
            noise_std: Standard deviation of Gaussian noise to inject during training.

        Returns:
            logit: Scalar logit for binary classification.
            cross_attention_weights: Cross-attention matrix of shape (P, I).
            turn_attention_weights: Turn-level attention weights of shape (P,).
        """
        # Inject embedding noise during training
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std
            interviewer_emb = interviewer_emb + torch.randn_like(interviewer_emb) * noise_std

        # Project both roles into shared space
        patient = self.projector(patient_emb)
        interviewer = self.projector(interviewer_emb)

        # Cross-role attention: patient queries interviewer
        context, cross_attn_weights = self.cross_attention(patient, interviewer)

        # Fuse patient + context with residual
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
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass for a padded batch of dual-role bags.

        Args:
            patient_bags: (batch_size, max_P, embedding_dim).
            interviewer_bags: (batch_size, max_I, embedding_dim).
            patient_sizes: Number of valid patient turns per bag.
            interviewer_sizes: Number of valid interviewer turns per bag.
            noise_std: Gaussian noise std dev applied to embeddings during training.

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

            logit, cross_attn, turn_attn = self.forward(
                patient_i, interviewer_i, noise_std=noise_std
            )

            logits.append(logit)
            cross_attention_list.append(cross_attn)
            turn_attention_list.append(turn_attn)

        logits = torch.stack(logits)
        return logits, cross_attention_list, turn_attention_list
