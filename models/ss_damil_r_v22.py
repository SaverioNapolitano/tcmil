import torch
import torch.nn as nn
from models.damil_r import CrossRoleAttention, RoleAwareFusion, AttentionPooling

class SSDamilRClassifierV22(nn.Module):
    """v22: Base DAMIL-R Architecture + Manifold Mixup API.
    
    This class recreates the highly robust SSDamilRClassifierV8_2 
    (single-head attention pooling, direct linear main head, no gated injection)
    which achieved the highest ROC-AUC baseline (v9a), but exposes the
    `forward_backbone`, `forward_batch_split`, and `forward_heads` APIs 
    necessary to perform Manifold Mixup during training (which achieved the
    highest F1 in v9d).
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 64,
        num_symptoms: int = 8,
        dropout_rate: float = 0.3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.working_dim = proj_dim if (proj_dim and proj_dim > 0) else embedding_dim

        # 1. Projector
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

        # 4. Single-Head Attention Pooling
        self.pooling = AttentionPooling(
            input_dim=self.working_dim,
            hidden_dim=32, # Matching baseline defaults
            temperature=temperature
        )

        # 5. Symptom Head (Auxiliary)
        self.symptom_head = nn.Sequential(
            nn.Linear(self.working_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_symptoms)
        )

        # 6. Simplified Main Head (Direct from Pool)
        self.main_classifier = nn.Linear(self.working_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> dict:
        pooled, _ = self.forward_backbone(patient_emb, interviewer_emb, noise_std)
        head_outputs = self.forward_heads(pooled)
        
        # In a non-batched pure forward pass, we might want the attention weights
        # So let's recompute or handle them properly if needed. But for training
        # mixup, we just need `forward_backbone` and `forward_heads`.
        
        return head_outputs

    def forward_backbone(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone (projector → cross-attention → fusion → pooling).

        Returns:
            pooled: Pooled bag representation of shape (D,).
            diversity_loss: Scalar zero (since v8.2 has single head pooling).
        """
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std
            interviewer_emb = interviewer_emb + torch.randn_like(interviewer_emb) * noise_std

        p_proj = self.projector(patient_emb)
        i_proj = self.projector(interviewer_emb)

        context, _ = self.cross_attention(p_proj, i_proj)
        instance_features = self.post_fusion_norm(self.fusion(p_proj, context))

        pooled, _ = self.pooling(instance_features)
        
        diversity_loss = torch.tensor(0.0, device=patient_emb.device)
        return pooled, diversity_loss

    def forward_heads(self, pooled: torch.Tensor) -> dict:
        """Run the classification heads on a (possibly mixed) pooled representation."""
        is_batched = pooled.dim() == 2

        sym_logits = self.symptom_head(pooled)
        main_h = self.dropout(pooled)
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
        """Run forward on a padded batch, returning stacked logits.

        This is the API expected by evaluate_v9 and train_epoch_v9.
        """
        batch_size = patient_bags.size(0)

        batch_main_logits = []
        batch_symptom_logits = []

        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]

            p_i = patient_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]

            res = self.forward(p_i, i_i, noise_std=noise_std)
            batch_main_logits.append(res["logit"])
            batch_symptom_logits.append(res["symptom_logits"])

        return {
            "logits": torch.stack(batch_main_logits),
            "symptom_logits": torch.stack(batch_symptom_logits),
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
