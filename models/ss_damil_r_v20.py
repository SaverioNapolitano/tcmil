"""SS-DAMIL-R v20: The Dual-Path Distillation Pipeline.

Incorporates Dual-Path Attention Pooling to capture both general context
(via attention-weighted mean pooling) and sudden extreme depressive indicators
(via feature-wise max pooling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.damil_r import CrossRoleAttention, RoleAwareFusion
from models.ss_damil_r import MultiHeadAttentionPooling


class DualPathPooling(nn.Module):
    """Combines Multi-Head Attention Pooling with Feature-wise Max Pooling.
    
    Args:
        input_dim: Dimensionality of instance embeddings.
        hidden_dim: Hidden dimension for attention scorer.
        n_heads: Number of attention heads.
        temperature: Softmax temperature.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, n_heads: int = 2, temperature: float = 1.0):
        super().__init__()
        self.attn_pooling = MultiHeadAttentionPooling(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            temperature=temperature
        )
        # We concatenate mean-pooled and max-pooled representations.
        # This doubles the dimension, so we project it back to input_dim.
        self.fusion_proj = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Compute dual-path pooling.
        
        Args:
            x: Tensor of shape (P, D).
            
        Returns:
            pooled: Merged representation of shape (D,).
            head_weights: List of attention weight tensors, each (P,).
            diversity_loss: Scalar measuring head similarity (to minimize).
        """
        attn_pooled, head_weights, diversity_loss = self.attn_pooling(x)
        
        # Max Pooling Path
        # x is (P, D)
        max_pooled, _ = torch.max(x, dim=0) # (D,)
        
        # Concatenate and Project
        combined = torch.cat([attn_pooled, max_pooled], dim=0) # (2D,)
        pooled = self.fusion_proj(combined) # (D,)
        
        return pooled, head_weights, diversity_loss


class SSDamilRClassifierV20(nn.Module):
    """v20: Dual-Path Pooling + Gated Symptom Injection.

    Architectural differences from V9d:
    - Replaces primitive MultiHeadAttentionPooling with DualPathPooling to
      prevent max-signal dilution.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 64,
        num_symptoms: int = 8,
        dropout_rate: float = 0.3,
        temperature: float = 1.0,
        n_pool_heads: int = 2,
    ):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.working_dim = proj_dim if (proj_dim and proj_dim > 0) else embedding_dim

        # 1. Projector (same as baseline)
        if proj_dim and proj_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.ReLU(),
            )
        else:
            self.projector = nn.Identity()

        # 2-3. Role Interaction (Backbone)
        self.cross_attention = CrossRoleAttention(dim=self.working_dim)
        self.fusion = RoleAwareFusion(dim=self.working_dim)
        self.post_fusion_norm = nn.LayerNorm(self.working_dim)

        # 4. Dual-Path Pooling (V20 Improvement)
        self.pooling = DualPathPooling(
            input_dim=self.working_dim,
            hidden_dim=32,
            n_heads=n_pool_heads,
            temperature=temperature,
        )

        # 5. Symptom Head (Auxiliary)
        self.symptom_head = nn.Sequential(
            nn.Linear(self.working_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_symptoms),
        )

        # 6. Gated Symptom Injection
        self.sym_projector = nn.Sequential(
            nn.Linear(num_symptoms, self.working_dim),
            nn.Tanh(),
        )
        self.inject_gate = nn.Linear(self.working_dim * 2, self.working_dim)
        nn.init.constant_(self.inject_gate.bias, 2.0)

        # 7. Main Classifier
        self.main_classifier = nn.Linear(self.working_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> dict:
        # Noise for stability
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std
            interviewer_emb = interviewer_emb + torch.randn_like(interviewer_emb) * noise_std

        # Feature Backbone
        p_proj = self.projector(patient_emb)
        i_proj = self.projector(interviewer_emb)

        context, _ = self.cross_attention(p_proj, i_proj)
        instance_features = self.post_fusion_norm(self.fusion(p_proj, context))

        # Dual-Path Pooling
        pooled, head_weights, diversity_loss = self.pooling(instance_features)

        # Auxiliary Symptom Predictions
        sym_logits = self.symptom_head(pooled)  # (8,)

        # Gated Symptom Injection
        sym_signal = self.sym_projector(sym_logits)  # (D,)
        gate_input = torch.cat([pooled, sym_signal], dim=0).unsqueeze(0)  # (1, 2D)
        gate = torch.sigmoid(self.inject_gate(gate_input)).squeeze(0)  # (D,)
        enriched = gate * pooled + (1.0 - gate) * sym_signal  # (D,)

        # Main Prediction
        main_h = self.dropout(enriched)
        main_logit = self.main_classifier(main_h).squeeze(-1)

        return {
            "logit": main_logit,
            "symptom_logits": sym_logits,       # (8,)
            "head_weights": head_weights,       # list of (P,) tensors
            "diversity_loss": diversity_loss,    # scalar
            "pooled_representation": pooled,    # (D,)
        }

    def forward_backbone(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std
            interviewer_emb = interviewer_emb + torch.randn_like(interviewer_emb) * noise_std

        p_proj = self.projector(patient_emb)
        i_proj = self.projector(interviewer_emb)

        context, _ = self.cross_attention(p_proj, i_proj)
        instance_features = self.post_fusion_norm(self.fusion(p_proj, context))

        pooled, _, diversity_loss = self.pooling(instance_features)
        return pooled, diversity_loss

    def forward_heads(self, pooled: torch.Tensor) -> dict:
        is_batched = pooled.dim() == 2

        sym_logits = self.symptom_head(pooled)
        sym_signal = self.sym_projector(sym_logits)

        if is_batched:
            gate_input = torch.cat([pooled, sym_signal], dim=1)  # (B, 2D)
        else:
            gate_input = torch.cat([pooled, sym_signal], dim=0).unsqueeze(0)  # (1, 2D)

        gate = torch.sigmoid(self.inject_gate(gate_input))  # (B, D) or (1, D)
        if not is_batched:
            gate = gate.squeeze(0)

        enriched = gate * pooled + (1.0 - gate) * sym_signal

        main_h = self.dropout(enriched)
        main_logit = self.main_classifier(main_h).squeeze(-1)

        key = "logits" if is_batched else "logit"
        return {key: main_logit, "symptom_logits": sym_logits}

    def forward_batch(
        self,
        patient_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_sizes: list[int],
        interviewer_sizes: list[int],
        noise_std: float = 0.0,
    ) -> dict:
        batch_size = patient_bags.size(0)

        batch_main_logits = []
        batch_symptom_logits = []
        batch_diversity_losses = []

        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]

            p_i = patient_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]

            res = self.forward(p_i, i_i, noise_std=noise_std)
            batch_main_logits.append(res["logit"])
            batch_symptom_logits.append(res["symptom_logits"])
            batch_diversity_losses.append(res["diversity_loss"])

        return {
            "logits": torch.stack(batch_main_logits),
            "symptom_logits": torch.stack(batch_symptom_logits),     # (B, 8)
            "diversity_loss": torch.stack(batch_diversity_losses).mean(),  # scalar
        }

    def forward_batch_split(
        self,
        patient_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_sizes: list[int],
        interviewer_sizes: list[int],
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = patient_bags.size(0)
        pooled_list = []
        div_losses = []

        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]
            p_i = patient_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]

            pooled, div_loss = self.forward_backbone(p_i, i_i, noise_std=noise_std)
            pooled_list.append(pooled)
            div_losses.append(div_loss)

        return torch.stack(pooled_list), torch.stack(div_losses).mean()
