"""Dialogue Mean baseline model for binary interview-level classification.

Architecture:
    1. Encode each utterance with a pretrained text encoder (RoBERTa).
    2. Extract the CLS token embedding per utterance.
    3. Mean-pool all utterance embeddings for an interview.
    4. Feed the pooled embedding through a small classification head.

For this version, the encoder is kept completely frozen. Utterance
embeddings are pre-computed once to radically speed up training of the
classification head.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def encode_utterances(
    utterances: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode a list of utterances and return CLS token embeddings.

    Args:
        utterances: List of utterance strings.
        tokenizer: Hugging Face tokenizer.
        model: Pretrained encoder (frozen, in eval mode).
        device: torch device.
        max_length: Max token length per utterance.
        batch_size: Encoding batch size.

    Returns:
        Tensor of shape (n_utterances, hidden_dim) with CLS embeddings.
    """
    all_embeddings = []

    for i in range(0, len(utterances), batch_size):
        batch = utterances[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # CLS token is at position 0
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)
        all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def mean_pool_interview(utterance_embeddings: torch.Tensor) -> torch.Tensor:
    """Mean-pool utterance embeddings into a single interview embedding.

    Args:
        utterance_embeddings: Tensor of shape (n_utterances, hidden_dim).

    Returns:
        Tensor of shape (hidden_dim,).
    """
    return utterance_embeddings.mean(dim=0)


class DialogueMeanClassifier(nn.Module):
    """Tiny classification head for binary prediction from pooled embeddings.

    Architecture: LayerNorm -> Dropout -> Linear(hidden_dim, 1)
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Interview embedding, shape (batch, hidden_dim).

        Returns:
            Logits, shape (batch, 1).
        """
        x = self.norm(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
