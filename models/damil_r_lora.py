"""DAMIL-R LoRA: Role-Aware Dual Attention MIL with Transformer Fine-Tuning.

This variant encapsulates the pre-trained encoder (MPNet) with LoRA adapters
to allow task-specific fine-tuning on clinical datasets.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

# Import base components from the verified damil_r implementation
from models.damil_r import CrossRoleAttention, RoleAwareFusion, AttentionPooling


class DAMILRLora(nn.Module):
    """DAMIL-R variant for end-to-end fine-tuning with LoRA.

    Encapsulates a Transformer encoder with LoRA adapters and the DAMIL-R
    role-aware layers. Designed for execution on clusters with high VRAM (24GB+).

    Args:
        encoder_name: Name of the pre-trained Transformer (e.g., all-mpnet-base-v2).
        lora_r: Rank of LoRA adapters.
        lora_alpha: Alpha scaling for LoRA.
        proj_dim: Projection dimension for the shared embedding space.
        att_hidden_dim: Hidden dimension for MIL attention pooling.
        dropout_rate: Dropout rate for turn-level representations.
        temperature: Initial temperature for attention pooling.
    """

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
        lora_r: int = 8,
        lora_alpha: int = 32,
        proj_dim: int = 64,
        att_hidden_dim: int = 32,
        dropout_rate: float = 0.3,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        # 1. Base Encoder
        base_model = AutoModel.from_pretrained(encoder_name)
        
        # 2. Apply LoRA
        # MPNet target modules: query, key, value projections in self-attention
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=None, # Feature extraction task
        )
        self.encoder = get_peft_model(base_model, peft_config)
        self.embedding_dim = base_model.config.hidden_size

        # 3. DAMIL-R Role-Aware Layers
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, proj_dim),
            nn.ReLU(),
        )
        self.cross_attention = CrossRoleAttention(dim=proj_dim)
        self.fusion = RoleAwareFusion(dim=proj_dim)
        self.attention_pooling = AttentionPooling(
            input_dim=proj_dim,
            hidden_dim=att_hidden_dim,
            temperature=temperature,
        )

        # 4. Final Head
        self.classifier = nn.Linear(proj_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def print_trainable_parameters(self):
        """Helper to verify LoRA is working as expected."""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def encode_bag(self, bag_input):
        """Pass a bag of utterances (P, SeqLen) through the LoRA encoder.
        
        Uses Mean Pooling over the token embeddings to produce turn vectors.
        """
        # input_ids: (P, L), attention_mask: (P, L)
        outputs = self.encoder(
            input_ids=bag_input["input_ids"],
            attention_mask=bag_input["attention_mask"]
        )
        token_embeddings = outputs.last_hidden_state # (P, L, D)
        
        # Mean Pooling (ignoring padding)
        input_mask_expanded = bag_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        patient_batch: dict,
        interviewer_batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with on-the-fly encoding.
        
        Args:
            patient_batch: Token IDs and masks for patient turns.
            interviewer_batch: Token IDs and masks for interviewer turns.
        """
        # 1. Encode turns on-the-fly (End-to-End gradients flow back through LoRA)
        p_emb = self.encode_bag(patient_batch)       # (P, D)
        i_emb = self.encode_bag(interviewer_batch)   # (I, D)

        # 2. Project into shared space
        p_proj = self.projector(p_emb)               # (P, d)
        i_proj = self.projector(i_emb)               # (I, d)

        # 3. DAMIL-R Interaction
        context, cross_attn_weights = self.cross_attention(p_proj, i_proj)
        fused = self.fusion(p_proj, context)         # (P, d)

        # 4. Pooling & Classification
        pooled, turn_attn_weights = self.attention_pooling(fused) # (d,)
        pooled = self.dropout(pooled)
        logit = self.classifier(pooled).squeeze(-1) # ()

        return logit, cross_attn_weights, turn_attn_weights
