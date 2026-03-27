"""DAMIL-H: Hierarchical Dual Attention MIL model for binary interview-level classification.

This model operates on pre-computed utterance embeddings (e.g., DistilBERT [CLS] vectors).
It applies a learned attention mechanism over utterances within each dialogue to produce
a single dialogue-level representation, which is then passed to a binary classifier.

Architecture:
    Input embeddings (batch, turns, 768)
    -> Linear projection
    -> Attention pooling over turns (softmax(v^T tanh(Wx)))
    -> Dropout
    -> Binary classifier head
"""

import torch
import torch.nn as nn


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


class DAMILHClassifier(nn.Module):
    """Hierarchical Dual Attention MIL classifier for depression detection.

    Takes pre-computed utterance embeddings for each dialogue, applies a linear
    projection, pools via learned attention over utterances, and predicts a
    binary logit.

    Args:
        embedding_dim: Dimension of input utterance embeddings (default: 768).
        proj_dim: Projection dimension. If None or 0, no projection is applied.
        att_hidden_dim: Hidden dimension for the attention scorer.
        dropout_rate: Dropout rate applied after attention pooling.
        temperature: Softmax temperature for attention weights.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int | None = None,
        att_hidden_dim: int = 32,
        dropout_rate: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()

        # Optional linear projection
        if proj_dim is not None and proj_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.ReLU(),
            )
            pooling_dim = proj_dim
        else:
            self.projector = nn.Identity()
            pooling_dim = embedding_dim

        # Attention pooling across utterances (turns)
        self.attention_pooling = AttentionPooling(
            input_dim=pooling_dim,
            hidden_dim=att_hidden_dim,
            temperature=temperature,
        )

        self.dropout = nn.Dropout(dropout_rate)

        # Binary classifier head
        self.classifier = nn.Linear(pooling_dim, 1)

    def forward(
        self,
        bag: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single bag (dialogue).

        Args:
            bag: Utterance embeddings of shape (num_turns, embedding_dim).
            mask: Optional boolean mask of shape (num_turns,).

        Returns:
            logit: Scalar logit for binary classification.
            attention_weights: Attention weights of shape (num_turns,).
        """
        # Project: (num_turns, pooling_dim)
        projected = self.projector(bag)

        # Attention pool: (pooling_dim,), (num_turns,)
        pooled, attention_weights = self.attention_pooling(projected, mask=mask)

        # Dropout + classify
        pooled = self.dropout(pooled)
        logit = self.classifier(pooled).squeeze(-1)

        return logit, attention_weights

    def forward_batch(
        self,
        bags: torch.Tensor,
        bag_sizes: list[int],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass for a padded batch of bags.

        Handles variable-length bags by iterating over each bag individually
        with masking. Compatible with batch_size >= 1.

        Args:
            bags: Padded tensor of shape (batch_size, max_turns, embedding_dim).
            bag_sizes: Number of valid (non-padding) turns per bag.

        Returns:
            logits: Tensor of shape (batch_size,).
            attention_weights_list: List of tensors, each (num_valid_turns,).
        """
        batch_size = bags.size(0)
        max_turns = bags.size(1)

        logits = []
        attention_weights_list = []

        for i in range(batch_size):
            n = bag_sizes[i]
            # Extract only valid turns (no padding)
            bag_i = bags[i, :n, :]  # (n, embedding_dim)
            logit, att_w = self.forward(bag_i)
            logits.append(logit)
            attention_weights_list.append(att_w)

        logits = torch.stack(logits)
        return logits, attention_weights_list
