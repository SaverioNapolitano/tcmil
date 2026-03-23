"""Flat MIL with attention pooling baseline model for binary interview-level classification."""

import torch
import torch.nn as nn
from transformers import AutoModel


class FlatMILAttention(nn.Module):
    """
    Flat MIL baseline with attention pooling.
    
    This model treats an interview as a bag of utterances. It encodes each utterance
    independently using a pretrained text encoder, optionally projects the embeddings,
    applies a learned attention mechanism over the utterances to compute attention weights,
    aggregates them via the attention weights, and finally predicts a binary logit for the bag.
    """
    def __init__(self, transformer_name: str, proj_dim: int | None = None, att_hidden_dim: int = 128):
        super().__init__()
        # Utterance encoder
        self.encoder = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Instance projection (optional)
        if proj_dim is not None:
            self.projector = nn.Sequential(
                nn.Linear(hidden_size, proj_dim),
                nn.ReLU()
            )
            agg_dim = proj_dim
        else:
            self.projector = nn.Identity()
            agg_dim = hidden_size
            
        # Attention MIL pooler
        self.attention_V = nn.Linear(agg_dim, att_hidden_dim)
        self.attention_w = nn.Linear(att_hidden_dim, 1, bias=False)
            
        # Bag classifier
        self.classifier = nn.Linear(agg_dim, 1)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        bag_sizes: list[int]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of shape [Total_Utterances, Seq_Len]
            attention_mask: Tensor of shape [Total_Utterances, Seq_Len]
            bag_sizes: List of integers indicating how many utterances belong to each bag in the batch.
            
        Returns:
            logits: Tensor of shape [Batch_Size], the predicted logits for each bag.
            attention_weights: List of Tensors of shape [Num_Utterances_in_Bag], attention scores for each bag.
        """
        # 1. Encode utterances
        # outputs.last_hidden_state: [Total_Utterances, Seq_Len, Hidden_Size]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # We use the representation of the [CLS] token (index 0)
        # cls_embeddings: [Total_Utterances, Hidden_Size]
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # 2. Instance projection
        # projected: [Total_Utterances, Agg_Dim]
        projected = self.projector(cls_embeddings)
        
        # 3. Attention MIL Bag Pooling
        # Split the flat batch back into individual bags
        bag_embeddings = torch.split(projected, bag_sizes)
        
        pooled_bags = []
        attention_weights_list = []
        for bag in bag_embeddings:
            # bag: [Num_Utterances_in_Bag, Agg_Dim]
            
            # Compute attention scores: w^T tanh(V * h)
            # a_scores: [Num_Utterances_in_Bag, 1]
            a_scores = self.attention_w(torch.tanh(self.attention_V(bag)))
            
            # a_weights: [Num_Utterances_in_Bag, 1]
            a_weights = torch.softmax(a_scores, dim=0)
            
            # apply attention weights: a_weights^T * bag -> [1, Agg_Dim]
            # using element-wise multiply and sum
            weighted_bag = (a_weights * bag).sum(dim=0) # [Agg_Dim]
            
            pooled_bags.append(weighted_bag)
            attention_weights_list.append(a_weights.squeeze(-1))
            
        # pooled: [Batch_Size, Agg_Dim]
        pooled = torch.stack(pooled_bags)
        
        # 4. Bag classifier
        # logits: [Batch_Size, 1] -> [Batch_Size]
        logits = self.classifier(pooled).squeeze(-1)
        
        return logits, attention_weights_list
