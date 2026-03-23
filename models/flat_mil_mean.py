"""Flat MIL with mean pooling baseline model for binary interview-level classification."""

import torch
import torch.nn as nn
from transformers import AutoModel


class FlatMILMeanPooling(nn.Module):
    """
    Flat MIL baseline with mean pooling.
    
    This model treats an interview as a bag of utterances. It encodes each utterance
    independently using a pretrained text encoder, optionally projects the embeddings,
    aggregates them via mean pooling, and finally predicts a binary logit for the bag.
    """
    def __init__(self, transformer_name: str, proj_dim: int | None = None):
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
            
        # Bag classifier
        self.classifier = nn.Linear(agg_dim, 1)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        bag_sizes: list[int]
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of shape [Total_Utterances, Seq_Len]
            attention_mask: Tensor of shape [Total_Utterances, Seq_Len]
            bag_sizes: List of integers indicating how many utterances belong to each bag in the batch.
            
        Returns:
            logits: Tensor of shape [Batch_Size], the predicted logits for each bag.
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
        
        # 3. Bag pooling (Mean Pooling)
        # Split the flat batch back into individual bags
        bag_embeddings = torch.split(projected, bag_sizes)
        
        pooled_bags = []
        for bag in bag_embeddings:
            # bag: [Num_Utterances_in_Bag, Agg_Dim]
            # mean_bag: [Agg_Dim]
            mean_bag = bag.mean(dim=0)
            pooled_bags.append(mean_bag)
            
        # pooled: [Batch_Size, Agg_Dim]
        pooled = torch.stack(pooled_bags)
        
        # 4. Bag classifier
        # logits: [Batch_Size, 1] -> [Batch_Size]
        logits = self.classifier(pooled).squeeze(-1)
        
        return logits
