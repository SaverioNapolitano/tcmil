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

    Supports both pre-computed embeddings and on-the-fly encoding with
    optional partial fine-tuning of the transformer encoder.

    Args:
        encoder: Pre-trained transformer model (e.g., DistilBertModel).
                 If None, the model expects pre-computed embeddings in forward().
        embedding_dim: Dimension of input utterance embeddings (default: 768).
        proj_dim: Projection dimension. If None or 0, no projection is applied.
        att_hidden_dim: Hidden dimension for the attention scorer.
        dropout_rate: Dropout rate applied after attention pooling.
        temperature: Softmax temperature for attention weights.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        embedding_dim: int = 768,
        proj_dim: int | None = None,
        att_hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim

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

    def unfreeze_top_n_layers(self, n: int):
        """Unfreeze the top N transformer layers and the pooler.
        
        For DistilBERT, layers are in self.encoder.transformer.layer.
        """
        if self.encoder is None:
            return

        # Start by freezing everything
        for p in self.encoder.parameters():
            p.requires_grad = False

        if n <= 0:
            return

        # Unfreeze top N layers
        # DistilBERT has 6 layers (0-5)
        if hasattr(self.encoder, "transformer"):
            layers = self.encoder.transformer.layer
            num_layers = len(layers)
            for i in range(num_layers - n, num_layers):
                for p in layers[i].parameters():
                    p.requires_grad = True
        
        # Also unfreeze the embeddings or other parts if requested, 
        # but here we stick to the top N transformer layers as requested.

    def forward(
        self,
        bag: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_tokenized: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single bag (dialogue).

        Args:
            bag: Either (num_turns, embedding_dim) embeddings 
                 OR (num_turns, seq_len) input_ids if is_tokenized=True.
            mask: Optional boolean mask of shape (num_turns,) for valid turns.
            is_tokenized: Whether the input bag contains raw tokens.
            attention_mask: (num_turns, seq_len) mask for the encoder if tokenized.

        Returns:
            logit: Scalar logit for binary classification.
            attention_weights: Attention weights of shape (num_turns,).
        """
        if is_tokenized:
            if self.encoder is None:
                raise ValueError("Model has no encoder but received tokenized input.")
            
            # bag: (num_turns, seq_len)
            # encoder output: (num_turns, seq_len, hidden_size)
            outputs = self.encoder(input_ids=bag, attention_mask=attention_mask)
            # use [CLS] token: (num_turns, hidden_size)
            x = outputs.last_hidden_state[:, 0, :]
        else:
            x = bag

        # Project: (num_turns, pooling_dim)
        projected = self.projector(x)

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
        is_tokenized: bool = False,
        attention_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass for a padded batch of bags.

        Args:
            bags: (batch_size, max_turns, embedding_dim) 
                  OR (batch_size, max_turns, seq_len) if tokenized.
            bag_sizes: Number of valid turns per bag.
            is_tokenized: Whether the input bags contain raw tokens.
            attention_masks: (batch_size, max_turns, seq_len) if tokenized.

        Returns:
            logits: Tensor of shape (batch_size,).
            attention_weights_list: List of tensors, each (num_valid_turns,).
        """
        batch_size = bags.size(0)
        logits = []
        attention_weights_list = []

        for i in range(batch_size):
            n = bag_sizes[i]
            bag_i = bags[i, :n, ...]  # (n, ...)
            
            if is_tokenized:
                attn_mask_i = attention_masks[i, :n, :] # (n, seq_len)
                logit, att_w = self.forward(
                    bag_i, is_tokenized=True, attention_mask=attn_mask_i
                )
            else:
                logit, att_w = self.forward(bag_i)
                
            logits.append(logit)
            attention_weights_list.append(att_w)

        logits = torch.stack(logits)
        return logits, attention_weights_list
