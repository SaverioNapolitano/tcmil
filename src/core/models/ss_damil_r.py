"""Symptom-Supervised DAMIL-R (SS-DAMIL-R) model family.

Three role-aware, symptom-supervised MIL classifiers, named after the
architectural feature that distinguishes each (matching the paper's labels):

    SSDamilRMH   (MH)   - multi-head attention pooling; the canonical
                          symptom-supervised model and legacy ceiling.
    SSDamilRGSI  (GSI)  - gated multi-head pooling with stochastic attention
                          dropout (instance masking) on top of MH.
    SSDamilRConv (Conv) - prepends a 1-D temporal convolution over the patient
                          sequence before cross-role attention.

All three share the DAMIL-R backbone (projector -> cross-role attention ->
role-aware fusion), an auxiliary 8-way symptom head, and gated symptom
injection into the main depression logit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.damil_r import CrossRoleAttention, RoleAwareFusion


class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention Pooling with Diversity Regularization.

    Each head produces an independent weighted sum over instances.
    A diversity loss encourages the heads to attend to different turns.

    Args:
        input_dim: Dimensionality of input embeddings.
        hidden_dim: Hidden dimension for each attention scorer.
        n_heads: Number of attention heads.
        temperature: Softmax temperature.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, n_heads: int = 2, temperature: float = 1.0):
        super().__init__()
        self.n_heads = n_heads
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

        # Independent attention scorers per head
        self.W = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(n_heads)])
        self.v = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(n_heads)])

        # Projection: concatenated heads -> working_dim
        self.merge = nn.Linear(input_dim * n_heads, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Compute multi-head attention-weighted pooling.

        Args:
            x: Tensor of shape (P, D).

        Returns:
            pooled: Merged representation of shape (D,).
            head_weights: List of attention weight tensors, each (P,).
            diversity_loss: Scalar measuring head similarity (to minimize).
        """
        head_pooled = []
        head_weights = []

        for i in range(self.n_heads):
            e = torch.tanh(self.W[i](x))  # (P, H)
            logits = self.v[i](e) / self.temperature  # (P, 1)
            weights = torch.softmax(logits, dim=0)  # (P, 1)
            pooled_i = torch.sum(x * weights, dim=0)  # (D,)
            head_pooled.append(pooled_i)
            head_weights.append(weights.squeeze(-1))  # (P,)

        # Merge heads: concat -> project
        merged = torch.cat(head_pooled, dim=0)  # (n_heads * D,)
        pooled = self.merge(merged)  # (D,)

        # Diversity loss: penalize similar attention distributions
        # cosine similarity between all pairs of head weights
        diversity_loss = torch.tensor(0.0, device=x.device)
        n_pairs = 0
        for i in range(self.n_heads):
            for j in range(i + 1, self.n_heads):
                cos_sim = F.cosine_similarity(
                    head_weights[i].unsqueeze(0),
                    head_weights[j].unsqueeze(0),
                )
                diversity_loss = diversity_loss + cos_sim.squeeze()
                n_pairs += 1
        if n_pairs > 0:
            diversity_loss = diversity_loss / n_pairs

        return pooled, head_weights, diversity_loss


class GatedMultiHeadAttentionPooling(nn.Module):
    """Gated Multi-Head Attention Pooling with Attention Dropout (Stochastic Instance Masking)."""
    def __init__(self, input_dim: int, hidden_dim: int = 32, n_heads: int = 2, temperature: float = 1.0, attn_dropout: float = 0.10):
        super().__init__()
        self.n_heads = n_heads
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.attn_dropout = attn_dropout

        self.V = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(n_heads)])
        self.U = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(n_heads)])
        self.w = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(n_heads)])

        for i in range(n_heads):
            nn.init.xavier_uniform_(self.V[i].weight)
            nn.init.xavier_uniform_(self.U[i].weight)
            nn.init.xavier_uniform_(self.w[i].weight)

        self.merge = nn.Linear(input_dim * n_heads, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        head_pooled = []
        head_weights = []

        for i in range(self.n_heads):
            v_x = torch.tanh(self.V[i](x))
            u_x = torch.sigmoid(self.U[i](x))
            e = v_x * u_x

            logits = self.w[i](e) / self.temperature

            if self.training and self.attn_dropout > 0.0:
                # Stochastic Instance Masking (Attention Dropout)
                # Drop logits with probability p by replacing with large negative value
                mask = torch.empty(logits.size(), device=logits.device).bernoulli_(self.attn_dropout).bool()
                # Ensure we don't mask everything out!
                if mask.all():
                    mask[0] = False
                logits = logits.masked_fill(mask, -1e9)

            weights = torch.softmax(logits, dim=0)

            pooled_i = torch.sum(x * weights, dim=0)
            head_pooled.append(pooled_i)
            head_weights.append(weights.squeeze(-1))

        merged = torch.cat(head_pooled, dim=0)
        pooled = self.merge(merged)

        # Diversity loss (cosine sim between heads)
        diversity_loss = torch.tensor(0.0, device=x.device)
        n_pairs = 0
        for i in range(self.n_heads):
            for j in range(i + 1, self.n_heads):
                cos_sim = F.cosine_similarity(
                    head_weights[i].unsqueeze(0),
                    head_weights[j].unsqueeze(0),
                )
                diversity_loss = diversity_loss + cos_sim.squeeze()
                n_pairs += 1
        if n_pairs > 0:
            diversity_loss = diversity_loss / n_pairs

        return pooled, head_weights, diversity_loss


class SSDamilRMH(nn.Module):
    """SS-DAMIL-R (MH): Multi-Head Pooling + Gated Symptom Injection.

    The canonical symptom-supervised model and legacy ceiling.

    Architecture:
    - Shared backbone: projector + cross-role attention + role-aware gated fusion.
    - Multi-Head Attention Pooling with diversity regularization (captures multiple
      interview perspectives: e.g. emotional content vs behavioral indicators).
    - Gated Symptom Injection: symptom predictions are projected into embedding
      space and gated with the pooled representation, allowing the model to
      optionally incorporate symptom signal for the main task.
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

        # 4. Multi-Head Attention Pooling
        self.pooling = MultiHeadAttentionPooling(
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
        # Project symptom logits (8,) -> embedding space (D,)
        self.sym_projector = nn.Sequential(
            nn.Linear(num_symptoms, self.working_dim),
            nn.Tanh(),
        )
        # Gate: decides how much symptom info to inject
        # Input: concat(pooled, sym_signal) -> (2D,) -> (D,)
        self.inject_gate = nn.Linear(self.working_dim * 2, self.working_dim)
        # Initialize bias to positive values so sigmoid(gate) ≈ 1.0 initially
        # This means the model starts by keeping the pooled representation (baseline behavior)
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

        # Multi-Head Pooling
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
        """Run the backbone (projector → cross-attention → fusion → pooling).

        Returns:
            pooled: Pooled bag representation of shape (D,).
            diversity_loss: Scalar diversity loss from multi-head pooling.
        """
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
        """Run the classification heads on a (possibly mixed) pooled representation.

        Args:
            pooled: Tensor of shape (D,) or (B, D).

        Returns:
            Dict with 'logit'/'logits' and 'symptom_logits'.
        """
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
        """Run backbone for each sample, return stacked pooled reps + mean diversity loss.

        Returns:
            pooled_batch: Tensor of shape (B, D).
            diversity_loss: Scalar mean diversity loss.
        """
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


class SSDamilRGSI(nn.Module):
    """SS-DAMIL-R (GSI): Gated Attention Pooling + Attention Dropout + Gated Symptom Injection.

    Replaces MH's multi-head pooling with a gated multi-head pooling that applies
    stochastic attention dropout (instance masking) for regularization; the
    gated symptom injection head is unchanged.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 64,
        num_symptoms: int = 8,
        dropout_rate: float = 0.3,
        temperature: float = 1.0,
        n_pool_heads: int = 2,
        attn_dropout: float = 0.10,
    ):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.working_dim = proj_dim if (proj_dim and proj_dim > 0) else embedding_dim

        if proj_dim and proj_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.ReLU(),
            )
        else:
            self.projector = nn.Identity()

        self.cross_attention = CrossRoleAttention(dim=self.working_dim)
        self.fusion = RoleAwareFusion(dim=self.working_dim)
        self.post_fusion_norm = nn.LayerNorm(self.working_dim)

        self.pooling = GatedMultiHeadAttentionPooling(
            input_dim=self.working_dim,
            hidden_dim=32,
            n_heads=n_pool_heads,
            temperature=temperature,
            attn_dropout=attn_dropout,
        )

        self.symptom_head = nn.Sequential(
            nn.Linear(self.working_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_symptoms),
        )

        self.sym_projector = nn.Sequential(
            nn.Linear(num_symptoms, self.working_dim),
            nn.Tanh(),
        )
        self.inject_gate = nn.Linear(self.working_dim * 2, self.working_dim)
        nn.init.constant_(self.inject_gate.bias, 2.0)

        self.main_classifier = nn.Linear(self.working_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, patient_emb: torch.Tensor, interviewer_emb: torch.Tensor, noise_std: float = 0.0) -> dict:
        pooled, diversity_loss = self.forward_backbone(patient_emb, interviewer_emb, noise_std)
        head_outputs = self.forward_heads(pooled)
        return {
            "logit": head_outputs["logits"] if "logits" in head_outputs else head_outputs["logit"],
            "symptom_logits": head_outputs["symptom_logits"],
            "diversity_loss": diversity_loss,
            "pooled_representation": pooled,
        }

    def forward_backbone(self, patient_emb: torch.Tensor, interviewer_emb: torch.Tensor, noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
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
            gate_input = torch.cat([pooled, sym_signal], dim=1)
        else:
            gate_input = torch.cat([pooled, sym_signal], dim=0).unsqueeze(0)

        gate = torch.sigmoid(self.inject_gate(gate_input))
        if not is_batched:
            gate = gate.squeeze(0)

        enriched = gate * pooled + (1.0 - gate) * sym_signal
        main_h = self.dropout(enriched)
        main_logit = self.main_classifier(main_h).squeeze(-1)

        key = "logits" if is_batched else "logit"
        return {key: main_logit, "symptom_logits": sym_logits}

    def forward_batch(self, patient_bags: torch.Tensor, interviewer_bags: torch.Tensor, patient_sizes: list[int], interviewer_sizes: list[int], noise_std: float = 0.0) -> dict:
        batch_size = patient_bags.size(0)
        batch_main_logits = []
        batch_symptom_logits = []
        batch_diversity_losses = []

        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]

            p_i = patient_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]

            pooled, div_loss = self.forward_backbone(p_i, i_i, noise_std=noise_std)
            head_outputs = self.forward_heads(pooled)

            batch_main_logits.append(head_outputs["logit"] if "logit" in head_outputs else head_outputs["logits"])
            batch_symptom_logits.append(head_outputs["symptom_logits"])
            batch_diversity_losses.append(div_loss)

        return {
            "logits": torch.stack(batch_main_logits),
            "symptom_logits": torch.stack(batch_symptom_logits),
            "diversity_loss": torch.stack(batch_diversity_losses).mean(),
        }

    def forward_batch_split(self, patient_bags: torch.Tensor, interviewer_bags: torch.Tensor, patient_sizes: list[int], interviewer_sizes: list[int], noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
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


class SSDamilRConv(nn.Module):
    """SS-DAMIL-R (Conv): Temporal Convolutional MIL.

    Adds a 1D Convolution over the patient sequence before the Cross-Role Attention
    and Multi-Head Attention Pooling. This natively captures chronological context
    (local sequence flow) with minimal parameters, avoiding the overfitting seen
    in Transformers on this small dataset.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 64,
        att_hidden_dim: int = 32,
        dropout_rate: float = 0.3,
        n_pool_heads: int = 2,
        conv_kernel_size: int = 3,
        num_symptoms: int = 8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proj_dim = proj_dim
        self.n_pool_heads = n_pool_heads
        self.num_symptoms = num_symptoms

        # 1. Projection (optional)
        self.has_proj = proj_dim > 0
        in_dim = embedding_dim
        if self.has_proj:
            self.p_proj = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.i_proj = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            in_dim = proj_dim

        # 2. Temporal Convolution over Patient Sequence
        self.temporal_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2  # Keep length same
        )
        self.temporal_norm = nn.LayerNorm(in_dim)
        self.temporal_act = nn.GELU()

        # 3. Cross-Role Attention (Patient -> Interviewer)
        self.cross_attention = CrossRoleAttention(dim=in_dim)
        self.fusion = RoleAwareFusion(dim=in_dim)
        self.post_fusion_norm = nn.LayerNorm(in_dim)

        # 4. Multi-Head Attention Pooling
        self.pooling = MultiHeadAttentionPooling(
            input_dim=in_dim,
            hidden_dim=att_hidden_dim,
            n_heads=n_pool_heads,
            temperature=1.0
        )

        # 5. Symptom Head
        self.symptom_head = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_symptoms),
        )

        # 6. Gated Symptom Injection
        self.sym_projector = nn.Sequential(
            nn.Linear(num_symptoms, in_dim),
            nn.Tanh(),
        )
        self.inject_gate = nn.Linear(in_dim * 2, in_dim)
        nn.init.constant_(self.inject_gate.bias, 2.0)

        # 7. Main Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.main_classifier = nn.Linear(in_dim, 1)

    def forward_backbone(
        self,
        patient_bag: torch.Tensor,
        interviewer_bag: torch.Tensor,
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns pooled representation and diversity loss."""
        # Optional noise
        if noise_std > 0.0 and self.training:
            patient_bag = patient_bag + torch.randn_like(patient_bag) * noise_std
            interviewer_bag = interviewer_bag + torch.randn_like(interviewer_bag) * noise_std

        # 1. Project
        if self.has_proj:
            p_feat = self.p_proj(patient_bag)
            i_feat = self.i_proj(interviewer_bag)
        else:
            p_feat = patient_bag
            i_feat = interviewer_bag

        # 2. Temporal Convolution
        # Conv1d expects (Batch, Channels, Length). Our bag is (Length, Channels).
        # We add dummy batch dim: (1, Length, Channels) -> permute -> (1, Channels, Length)
        p_feat_batch = p_feat.unsqueeze(0).permute(0, 2, 1)
        conv_out = self.temporal_conv(p_feat_batch)
        # Permute back and squeeze
        conv_out = conv_out.permute(0, 2, 1).squeeze(0)

        # Residual connection + Norm + Activation
        p_feat = self.temporal_norm(p_feat + self.temporal_act(conv_out))

        # 3. Cross-Attention
        context, _ = self.cross_attention(p_feat, i_feat)
        fused = self.post_fusion_norm(self.fusion(p_feat, context))

        # 4. Pooling
        pooled, _, div_loss = self.pooling(fused)

        return pooled, div_loss

    def forward_heads(self, pooled: torch.Tensor) -> dict:
        """Returns logits dict from pooled representation. Works on batched (B, D) or single (D,)."""
        sym_logits = self.symptom_head(pooled)
        sym_signal = self.sym_projector(sym_logits)

        gate_input = torch.cat([pooled, sym_signal], dim=-1)
        gate = torch.sigmoid(self.inject_gate(gate_input))
        enriched = gate * pooled + (1.0 - gate) * sym_signal

        main_h = self.dropout(enriched)
        main_logit = self.main_classifier(main_h).squeeze(-1)

        return {
            "logits": main_logit,
            "symptom_logits": sym_logits,
        }

    def forward_batch(
        self,
        patient_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_sizes: list[int],
        interviewer_sizes: list[int],
        noise_std: float = 0.0,
    ) -> dict:
        """Sequential processing over batch elements."""
        batch_size = patient_bags.size(0)
        batch_main = []
        batch_sym = []
        batch_div = []

        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]

            p_i = patient_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]

            pooled, div_loss = self.forward_backbone(p_i, i_i, noise_std=noise_std)
            head_out = self.forward_heads(pooled.unsqueeze(0))

            batch_main.append(head_out["logits"])
            batch_sym.append(head_out["symptom_logits"])
            batch_div.append(div_loss)

        return {
            "logits": torch.cat(batch_main),
            "symptom_logits": torch.cat(batch_sym),
            "diversity_loss": torch.stack(batch_div).mean(),
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
