import torch
import torch.nn as nn
import torch.nn.functional as F

from models.damil_r import CrossRoleAttention, RoleAwareFusion, AttentionPooling


class PrototypicalAttentionPooling(nn.Module):
    """Symptom-Specific + Global Prototypical Attention Pooling.
    
    Computes similarities between utterances and N symptom prototypes 
    PLUS one global "depression" prototype.
    """
    def __init__(self, input_dim: int, num_symptoms: int = 8, scale: float = 10.0):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.num_prototypes = num_symptoms + 1 # +1 for Global
        
        # Learnable scale
        self.scale = nn.Parameter(torch.tensor(scale))
        
        # Learnable prototypes
        # Index 0: Global Depression Prototype
        # Index 1-8: Symptom-Specific Prototypes
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, input_dim))
        nn.init.orthogonal_(self.prototypes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (BagSize, D) - Instance features
        Returns:
            pooled: (9, D) - 1 Global + 8 Symptom representations
            attn_weights: (9, BagSize)
        """
        # Stabilized Similarity: Cosine Similarity with learnable scale
        p_norm = F.normalize(self.prototypes, p=2, dim=1, eps=1e-8)
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        
        # scores: (9, BagSize)
        scores = torch.mm(p_norm, x_norm.transpose(0, 1)) * self.scale
        scores = torch.clamp(scores, min=-50, max=50) # Prevent softmax overflow
        
        # Softmax over BagSize
        attn_weights = F.softmax(scores, dim=1) # (9, BagSize)
        
        # Pool: (9, BagSize) @ (BagSize, D) -> (9, D)
        pooled = torch.mm(attn_weights, x)
        
        return pooled, attn_weights


class SSDamilRClassifier(nn.Module):
    """v6: Dynamic Symptom-Aggregated MIL Model with Global Prototype.
    
    Uses binary classification for symptoms (presence/absence).
    Architecture:
    - Backbone: DAMIL-R (Cross-Role Attention + Gated Fusion)
    - Pooling: 9 Prototypes (1 Global + 8 Symptoms)
    - Symptom Aggregation: Learned Attention over 8 symptom representations
    - Main Fusion: Gated Residual combination of Global + Aggregated symptoms
    - Aux Heads: 8 Binary Symptom Classifiers
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 128,
        num_symptoms: int = 8,
        dropout_rate: float = 0.4,
        scale: float = 10.0,
    ):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.working_dim = proj_dim if proj_dim > 0 else embedding_dim

        # 1. Projector
        if proj_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.ReLU(),
            )
        else:
            self.projector = nn.Identity()

        # 2-3. Role Interaction (DAMIL-R Backbone)
        self.cross_attention = CrossRoleAttention(dim=self.working_dim)
        self.fusion = RoleAwareFusion(dim=self.working_dim)

        # 4. Prototypical Attention Pooling (Global + Symptoms)
        self.pooling = PrototypicalAttentionPooling(
            input_dim=self.working_dim,
            num_symptoms=num_symptoms,
            scale=scale
        )
        self.post_fusion_norm = nn.LayerNorm(self.working_dim)

        # 5. Symptom Heads (Binary classification)
        self.symptom_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.working_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1) # Binary: presence vs absence
            ) for _ in range(num_symptoms)
        ])

        # 6. Symptom Aggregation (Learned Attention)
        self.sym_attn_v = nn.Linear(self.working_dim, 1)
        
        # 7. Main Fusion (Gated Residual)
        self.main_gate = nn.Linear(self.working_dim * 2, self.working_dim)
        self.main_transform = nn.Linear(self.working_dim * 2, self.working_dim)
        self.main_norm = nn.LayerNorm(self.working_dim)
        
        # 8. Main Head
        self.main_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.working_dim, 1)
        )

    def forward(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> dict:
        # Noise for stability
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std

        # Feature Backbone
        p_proj = self.projector(patient_emb)
        i_proj = self.projector(interviewer_emb)
        
        context, _ = self.cross_attention(p_proj, i_proj)
        instance_features = self.post_fusion_norm(self.fusion(p_proj, context))

        # Prototypical Pooling
        # pooled: (9, Dim), turn_attn: (9, BagSize)
        pooled, turn_attn = self.pooling(instance_features)

        # Symptom Predictions (Binary)
        # We use pooled[1:] for the 8 symptoms
        symptom_logits = []
        for i in range(self.num_symptoms):
            s_out = self.symptom_heads[i](pooled[i+1])
            symptom_logits.append(s_out)
        symptom_logits = torch.cat(symptom_logits) # (8,)

        # Main Depression Prediction
        # 1. Aggregate Symptom representations via attention
        sym_prototypes = pooled[1:] # (8, D)
        sym_attn_logits = self.sym_attn_v(sym_prototypes).squeeze(-1) # (8,)
        sym_weights = torch.softmax(sym_attn_logits, dim=0)
        symptom_repr = torch.sum(sym_prototypes * sym_weights.unsqueeze(-1), dim=0) # (D,)
        
        # 2. Gated Fusion: Global + Symptom
        global_repr = pooled[0]
        combined = torch.cat([global_repr, symptom_repr], dim=0).unsqueeze(0) # (1, 2D)
        
        z = torch.sigmoid(self.main_gate(combined))
        h_tilde = torch.relu(self.main_transform(combined))
        main_h = self.main_norm(z * global_repr.unsqueeze(0) + (1.0 - z) * h_tilde).squeeze(0)
        
        main_logit = self.main_classifier(main_h).squeeze(-1)

        return {
            "logit": main_logit,
            "symptom_logits": symptom_logits,  # (8,)
            "turn_attention": turn_attn,       # (9, BagSize)
            "prototypes": self.pooling.prototypes
        }

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
        batch_prototypes = None
        
        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]
            
            p_i = patient_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]
            
            res = self.forward(p_i, i_i, noise_std=noise_std)
            batch_main_logits.append(res["logit"])
            batch_symptom_logits.append(res["symptom_logits"])
            if batch_prototypes is None:
                batch_prototypes = res["prototypes"]
            
        return {
            "logits": torch.stack(batch_main_logits),
            "symptom_logits": torch.stack(batch_symptom_logits), # (B, 8)
            "prototypes": batch_prototypes # (9, D)
        }


class SymptomExpertAttentionPooling(nn.Module):
    """9-Head Symptom-Specific Attention Pooling.
    
    Each head is an independent "expert" for a specific symptom (8) 
    plus one "residual/general" head (1).
    """
    def __init__(self, input_dim: int, num_experts: int = 9, hidden_dim: int = 32):
        super().__init__()
        self.num_experts = num_experts
        
        # Independent attention scorers for each expert
        self.W = nn.Parameter(torch.randn(num_experts, input_dim, hidden_dim))
        self.v = nn.Parameter(torch.randn(num_experts, hidden_dim, 1))
        
        # Initialization
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (BagSize, D) - Instance features
        Returns:
            pooled: (9, D) - 9 Expert representations
            attn_weights: (9, BagSize)
        """
        # x: (BagSize, D)
        # self.W: (9, D, H)
        # e: (9, BagSize, H)
        e = torch.tanh(torch.matmul(x.unsqueeze(0), self.W))
        
        # attn_logits: (9, BagSize, 1)
        attn_logits = torch.matmul(e, self.v)
        
        # attn_weights: (9, BagSize)
        attn_weights = F.softmax(attn_logits.squeeze(-1), dim=1)
        
        # pooled: (9, BagSize) @ (BagSize, D) -> (9, D)
        pooled = torch.matmul(attn_weights, x)
        
        return pooled, attn_weights


class SSDamilRClassifierV7(nn.Module):
    """v7: Symptom-Expert Multi-Head Attention Model.
    
    Architecture:
    - Backbone: DAMIL-R (Cross-Role Attention + Gated Fusion)
    - Pooling: 9 Symptom Experts (8 PHQ Symptoms + 1 Residual)
    - Symptom Heads: 8 Binary Predictors (Head_i -> Symptom_i)
    - Symptom-Gated Fusion: Final prediction is weighted sum of experts
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 128,
        num_symptoms: int = 8,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.working_dim = proj_dim if proj_dim > 0 else embedding_dim

        # 1. Projector
        if proj_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.ReLU(),
            )
        else:
            self.projector = nn.Identity()

        # 2-3. Role Interaction (DAMIL-R Backbone)
        self.cross_attention = CrossRoleAttention(dim=self.working_dim)
        self.fusion = RoleAwareFusion(dim=self.working_dim)

        # 4. Symptom Expert Pooling (9 heads)
        self.pooling = SymptomExpertAttentionPooling(
            input_dim=self.working_dim,
            num_experts=num_symptoms + 1,
            hidden_dim=32
        )
        self.post_fusion_norm = nn.LayerNorm(self.working_dim)

        # 5. Symptom Predictors (Expert_i -> Symptom_i)
        self.symptom_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.working_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ) for _ in range(num_symptoms)
        ])

        # 6. Final Head
        self.main_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.working_dim, 1)
        )

    def forward(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> dict:
        # Noise for stability
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std

        # Feature Backbone
        p_proj = self.projector(patient_emb)
        i_proj = self.projector(interviewer_emb)
        
        context, _ = self.cross_attention(p_proj, i_proj)
        instance_features = self.post_fusion_norm(self.fusion(p_proj, context))

        # Symptom Expert Pooling
        # pooled_experts: (9, Dim), expert_attn: (9, BagSize)
        pooled_experts, expert_attn = self.pooling(instance_features)

        # Symptom Predictions (Experts 0-7)
        symptom_logits = []
        for i in range(self.num_symptoms):
            s_out = self.symptom_predictors[i](pooled_experts[i])
            symptom_logits.append(s_out)
        symptom_logits = torch.cat(symptom_logits) # (8,)

        # Symptom-Gated Fusion
        # w_i = sigmoid(s_i)
        w = torch.sigmoid(symptom_logits) # (8,)
        
        # Combine Experts: Sum(w_i * Head_i) + Head_Residual
        # (8, 1) * (8, D) -> (8, D)
        gated_experts = pooled_experts[:self.num_symptoms] * w.unsqueeze(-1)
        residual_expert = pooled_experts[self.num_symptoms] # (D,)
        
        # Aggregated Representation
        final_repr = torch.sum(gated_experts, dim=0) + residual_expert
        
        main_logit = self.main_classifier(final_repr).squeeze(-1)

        return {
            "logit": main_logit,
            "symptom_logits": symptom_logits,  # (8,)
            "expert_attention": expert_attn,   # (9, BagSize)
            "expert_representations": pooled_experts # (9, D)
        }

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
        batch_experts = []
        
        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]
            
            p_i = patient_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]
            
            res = self.forward(p_i, i_i, noise_std=noise_std)
            batch_main_logits.append(res["logit"])
            batch_symptom_logits.append(res["symptom_logits"])
            batch_experts.append(res["expert_representations"])
            
        return {
            "logits": torch.stack(batch_main_logits),
            "symptom_logits": torch.stack(batch_symptom_logits), # (B, 8)
            "expert_representations": torch.stack(batch_experts) # (B, 9, D)
        }


class SSDamilRClassifierV8(nn.Module):
    """v8: Shared-Bottom Multi-Task Model with Symptom-Aware Fusion.
    
    Architecture:
    - Backbone: DAMIL-R (Cross-Role Attention + Gated Fusion)
    - Pooling: Single-Head Attention Pooling (STABLE)
    - Symptom Head: MLP predicting all 8 symptoms from pooled representation
    - Main Head: MLP taking concat(pooled_repr, symptom_logits)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 128,
        num_symptoms: int = 8,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.working_dim = proj_dim if proj_dim > 0 else embedding_dim

        # 1. Projector
        if proj_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.ReLU(),
            )
        else:
            self.projector = nn.Identity()

        # 2-3. Role Interaction (DAMIL-R Backbone)
        self.cross_attention = CrossRoleAttention(dim=self.working_dim)
        self.fusion = RoleAwareFusion(dim=self.working_dim)

        # 4. Single-Head Attention Pooling
        self.pooling = AttentionPooling(
            input_dim=self.working_dim,
            hidden_dim=64,
            temperature=1.0
        )
        self.post_fusion_norm = nn.LayerNorm(self.working_dim)

        # 5. Symptom Head (Predicts all 8 from shared pool)
        self.symptom_head = nn.Sequential(
            nn.Linear(self.working_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_symptoms)
        )

        # 6. Main classifier (Concatenated Input)
        # Takes Pooled Repr (D) + Symptom Logits (8)
        self.main_classifier = nn.Sequential(
            nn.Linear(self.working_dim + num_symptoms, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> dict:
        # Noise for stability
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std

        # Feature Backbone
        p_proj = self.projector(patient_emb)
        i_proj = self.projector(interviewer_emb)
        
        context, _ = self.cross_attention(p_proj, i_proj)
        instance_features = self.post_fusion_norm(self.fusion(p_proj, context))

        # Single-Head Pooling
        pooled, attn_weights = self.pooling(instance_features)

        # Symptom Predictions
        sym_logits = self.symptom_head(pooled) # (8,)

        # Symptom-Aware Fusion for Depression
        # concat pooled (D) + logits (8)
        combined = torch.cat([pooled, sym_logits], dim=0)
        main_logit = self.main_classifier(combined).squeeze(-1)

        return {
            "logit": main_logit,
            "symptom_logits": sym_logits,      # (8,)
            "turn_attention": attn_weights,    # (BagSize,)
            "pooled_representation": pooled    # (D,)
        }

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
        batch_pooled = []
        
        for i in range(batch_size):
            p_n = patient_sizes[i]
            i_n = interviewer_sizes[i]
            
            p_i = patient_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]
            
            res = self.forward(p_i, i_i, noise_std=noise_std)
            batch_main_logits.append(res["logit"])
            batch_symptom_logits.append(res["symptom_logits"])
            batch_pooled.append(res["pooled_representation"])
            
        return {
            "logits": torch.stack(batch_main_logits),
            "symptom_logits": torch.stack(batch_symptom_logits), # (B, 8)
            "pooled_representations": torch.stack(batch_pooled)  # (B, D)
        }


class SSDamilRClassifierV8_2(nn.Module):
    """v8.2: Shared-Bottom MTL with Simplified Main Head (Baseline Recovery).
    
    This version reverts the main classification head to a single linear layer,
    similar to the baseline DAMIL-R, to prevent overfitting on small datasets.
    The symptom classification is kept as a pure auxiliary supervision task.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 128,
        num_symptoms: int = 8,
        dropout_rate: float = 0.3,
        temperature: float = 1.0,
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

        # 4. Single-Head Attention Pooling
        self.pooling = AttentionPooling(
            input_dim=self.working_dim,
            hidden_dim=32, # Matching baseline defaults
            temperature=temperature
        )
        self.post_fusion_norm = nn.LayerNorm(self.working_dim)

        # 5. Symptom Head (Auxiliary)
        self.symptom_head = nn.Sequential(
            nn.Linear(self.working_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_symptoms)
        )

        # 6. Simplified Main Head (Direct from Pool)
        # Matches DAMILRClassifier head complexity
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

        # Pooling
        pooled, attn_weights = self.pooling(instance_features)

        # Auxiliary Symptom Predictions
        sym_logits = self.symptom_head(pooled) # (8,)

        # Main Prediction (Pure Baseline Style)
        # We dropout BEFORE the linear layer to match damil_r.py:L263-264
        main_h = self.dropout(pooled)
        main_logit = self.main_classifier(main_h).squeeze(-1)

        return {
            "logit": main_logit,
            "symptom_logits": sym_logits,   # (8,)
            "turn_attention": attn_weights, # (BagSize,)
            "pooled_representation": pooled # (D,)
        }

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
            "symptom_logits": torch.stack(batch_symptom_logits), # (B, 8)
        }


class MultiHeadAttentionPooling(nn.Module):
    """2-Head Attention Pooling with Diversity Regularization.

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


class SSDamilRClassifierV9(nn.Module):
    """v9: Multi-Head Pooling + Gated Symptom Injection.

    Architectural differences from v8.2:
    - 2-Head Attention Pooling with diversity regularization (captures multiple
      interview perspectives: e.g. emotional content vs behavioral indicators)
    - Gated Symptom Injection: symptom predictions are projected into embedding
      space and gated with the pooled representation, allowing the model to
      optionally incorporate symptom signal for the main task.
    - Shared backbone (projector + cross-role attention + gated fusion) unchanged.
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
