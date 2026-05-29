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

        # Xavier initialization (restored from default)
        # for i in range(n_heads):
        #     nn.init.xavier_uniform_(self.W[i].weight)
        #     nn.init.xavier_uniform_(self.v[i].weight)

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


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for variable-length sequences.

    Injects relative turn-order information into instance features before
    attention pooling, enabling the model to capture the temporal structure
    of clinical interviews (icebreakers → symptom probing → personal questions).

    Args:
        d_model: Dimensionality of input embeddings.
        max_len: Maximum sequence length to support.
        scale: Scaling factor for positional signal (small to avoid
               overwhelming semantic content).
    """

    def __init__(self, d_model: int, max_len: int = 500, scale: float = 0.1):
        super().__init__()
        self.scale = scale

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # handle odd d_model
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add scaled positional encoding to input.

        Args:
            x: Tensor of shape (seq_len, d_model).

        Returns:
            Tensor of shape (seq_len, d_model) with positional info added.
        """
        seq_len = x.size(0)
        return x + self.scale * self.pe[:seq_len]


class SSDamilRClassifierV11c(nn.Module):
    """v11c: Temporal-Aware Multi-Head Pooling + Gated Symptom Injection.

    Identical to SSDamilRClassifierV9 but adds sinusoidal positional encoding
    to instance features before attention pooling, enabling the model to
    exploit the temporal structure of clinical interviews.

    Architectural differences from V9:
    - PositionalEncoding applied to fused instance features before pooling
    - All other components (backbone, pooling, symptom injection) unchanged
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 64,
        num_symptoms: int = 8,
        dropout_rate: float = 0.3,
        temperature: float = 1.0,
        n_pool_heads: int = 2,
        pe_scale: float = 0.1,
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

        # 3.5 Positional Encoding (NEW in v11c)
        self.pos_encoding = PositionalEncoding(
            d_model=self.working_dim,
            max_len=500,
            scale=pe_scale,
        )

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

        # v11c: Add positional encoding BEFORE pooling
        instance_features = self.pos_encoding(instance_features)

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
        """Run the backbone (projector → cross-attention → fusion → PE → pooling).

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

        # v11c: Add positional encoding
        instance_features = self.pos_encoding(instance_features)

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


class SSDamilRClassifierV12(nn.Module):
    """v12: Symptom-Guided Cross-Attention + MR-Drop.

    Architectural differences from V9/11b:
    - Replaces the linear gated symptom injection with Symptom-Guided Cross-Attention.
    - 8 trainable symptom embeddings are scaled by their predicted probability.
    - The pooled representation uses Multi-Head Attention to selectively attend 
      to the active symptoms.
    - Shared backbone (projector + cross-role attention + gated fusion + pooling) unchanged.
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

        # 6. Symptom-Guided Cross-Attention
        # Trainable symptom embeddings (one for each symptom)
        self.sym_embeddings = nn.Parameter(torch.randn(num_symptoms, self.working_dim))
        nn.init.xavier_uniform_(self.sym_embeddings)

        # Multihead Attention: Query=Pooled, Key=Value=Scaled Symptoms
        self.sym_cross_attn = nn.MultiheadAttention(
            embed_dim=self.working_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.post_attn_norm = nn.LayerNorm(self.working_dim)

        # 7. Main Classifier
        self.main_classifier = nn.Linear(self.working_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> dict:
        # This mirrors forward_backbone + forward_heads
        pooled, diversity_loss = self.forward_backbone(patient_emb, interviewer_emb, noise_std)
        head_outputs = self.forward_heads(pooled)
        return {
            "logit": head_outputs["logits"] if "logits" in head_outputs else head_outputs["logit"],
            "symptom_logits": head_outputs["symptom_logits"],
            "diversity_loss": diversity_loss,
            "pooled_representation": pooled,
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
        
        sym_logits = self.symptom_head(pooled) # (B, 8) or (8,)
        sym_probs = torch.sigmoid(sym_logits)

        # Prepare Query: (B, 1, D)
        query = pooled.unsqueeze(1) if is_batched else pooled.unsqueeze(0).unsqueeze(0)

        # Prepare Key/Value: Scale symptom embeddings by their predicted probability
        if is_batched:
            # sym_probs: (B, 8), sym_embeddings: (8, D)
            # scale: (B, 8, 1) * (1, 8, D) => (B, 8, D)
            kv = sym_probs.unsqueeze(-1) * self.sym_embeddings.unsqueeze(0)
        else:
            # sym_probs: (8,), sym_embeddings: (8, D)
            kv = sym_probs.unsqueeze(-1) * self.sym_embeddings
            kv = kv.unsqueeze(0) # (1, 8, D)

        # Cross Attention
        # query: (B, 1, D)
        # key = value: (B, 8, D)
        attn_out, _ = self.sym_cross_attn(query, kv, kv) # (B, 1, D)
        
        # Residual and Norm
        enriched = query + attn_out
        enriched = self.post_attn_norm(enriched).squeeze(1) # (B, D)

        if not is_batched:
            enriched = enriched.squeeze(0)

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
            "symptom_logits": torch.stack(batch_symptom_logits),     
            "diversity_loss": torch.stack(batch_diversity_losses).mean(),
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


class GatedMultiHeadAttentionPooling(nn.Module):
    """Gated Multi-Head Attention Pooling (based on Ilse et al., 2018).
    
    Replaces standard tanh attention with a more expressive formulation:
    weights = softmax(w^T (tanh(V*x) * sigmoid(U*x)))
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, n_heads: int = 2, temperature: float = 1.0):
        super().__init__()
        self.n_heads = n_heads
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

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
            weights = torch.softmax(logits, dim=0)
            
            pooled_i = torch.sum(x * weights, dim=0)
            head_pooled.append(pooled_i)
            head_weights.append(weights.squeeze(-1))

        merged = torch.cat(head_pooled, dim=0)
        pooled = self.merge(merged)

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


class SSDamilRClassifierV13(nn.Module):
    """v13: Temporal Context (BiGRU) + Gated Attention Pooling + Symptom Injection.
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

        # Temporal Context via lightweight BiGRU to prevent overfitting
        self.temporal_context = nn.GRU(
            input_size=self.working_dim,
            hidden_size=self.working_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.pooling = GatedMultiHeadAttentionPooling(
            input_dim=self.working_dim,
            hidden_dim=32,
            n_heads=n_pool_heads,
            temperature=temperature,
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

        # Add unsqueeze for nn.GRU (batch_size, seq_len, dim)
        instance_features = instance_features.unsqueeze(0)
        instance_features, _ = self.temporal_context(instance_features)
        instance_features = instance_features.squeeze(0)

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


class SSDamilRClassifierV14(nn.Module):
    """v14: Dialogue-Aware Depression Detection.

    Identical to SSDamilRClassifierV9 but configured for:
    - Larger encoder output (1024-dim from BAAI/bge-large-en-v1.5)
    - Larger projection dimension (128) to leverage richer features
    - Q-A dialogue pair embeddings as patient instances

    The architecture is deliberately kept very close to the proven v9d
    configuration. The performance improvement comes from the input quality
    (dialogue-pair preprocessing + upgraded encoder), not from architectural
    changes.

    Architecture:
    - Backbone: DAMIL-R (Cross-Role Attention + Gated Fusion)
    - Pooling: Multi-Head Attention Pooling (2 heads) with diversity loss
    - Symptom Head: MLP predicting 8 PHQ symptoms (auxiliary)
    - Gated Symptom Injection: symptom predictions projected and gated
    - Main Head: Linear classifier
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        proj_dim: int = 128,
        num_symptoms: int = 8,
        dropout_rate: float = 0.3,
        temperature: float = 1.0,
        n_pool_heads: int = 2,
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

        # 4. Multi-Head Attention Pooling
        self.pooling = MultiHeadAttentionPooling(
            input_dim=self.working_dim,
            hidden_dim=64,  # Slightly larger hidden dim for richer features
            n_heads=n_pool_heads,
            temperature=temperature,
        )

        # 5. Symptom Head (Auxiliary)
        self.symptom_head = nn.Sequential(
            nn.Linear(self.working_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_symptoms),
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

        # Multi-Head Pooling
        pooled, head_weights, diversity_loss = self.pooling(instance_features)

        # Auxiliary Symptom Predictions
        sym_logits = self.symptom_head(pooled)

        # Gated Symptom Injection
        sym_signal = self.sym_projector(sym_logits)
        gate_input = torch.cat([pooled, sym_signal], dim=0).unsqueeze(0)
        gate = torch.sigmoid(self.inject_gate(gate_input)).squeeze(0)
        enriched = gate * pooled + (1.0 - gate) * sym_signal

        # Main Prediction
        main_h = self.dropout(enriched)
        main_logit = self.main_classifier(main_h).squeeze(-1)

        return {
            "logit": main_logit,
            "symptom_logits": sym_logits,
            "head_weights": head_weights,
            "diversity_loss": diversity_loss,
            "pooled_representation": pooled,
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
            "symptom_logits": torch.stack(batch_symptom_logits),
            "diversity_loss": torch.stack(batch_diversity_losses).mean(),
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


class SetTransformerBlock(nn.Module):
    """Permutation-invariant Self-Attention block for global context."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (seq_len, dim) or (batch, seq_len, dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            sq = True
        else:
            sq = False
            
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        if sq:
            x = x.squeeze(0)
        return x


class SSDamilRClassifierV16(nn.Module):
    """v16: Set-Transformer DAMIL-R.

    Architectural redesign:
    - Preprocessing gives sliding-window local context.
    - Set-Transformer block replaces the BiGRU to allow global cross-utterance 
      context without overfitting to sequence rigidity. No Positional Encodings.
    - Preserves Multi-Head Attention Pooling and Gated Symptom Injection.
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

        # 4. Set Transformer Block (Global Context)
        self.set_transformer = SetTransformerBlock(
            dim=self.working_dim, 
            num_heads=4, 
            dropout=0.1
        )

        # 5. Multi-Head Attention Pooling
        self.pooling = MultiHeadAttentionPooling(
            input_dim=self.working_dim,
            hidden_dim=32,
            n_heads=n_pool_heads,
            temperature=temperature,
        )

        # 6. Symptom Head (Auxiliary)
        self.symptom_head = nn.Sequential(
            nn.Linear(self.working_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_symptoms),
        )

        # 7. Gated Symptom Injection
        self.sym_projector = nn.Sequential(
            nn.Linear(num_symptoms, self.working_dim),
            nn.Tanh(),
        )
        self.inject_gate = nn.Linear(self.working_dim * 2, self.working_dim)
        nn.init.constant_(self.inject_gate.bias, 2.0)

        # 8. Main Classifier
        self.main_classifier = nn.Linear(self.working_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> dict:
        pooled, diversity_loss = self.forward_backbone(patient_emb, interviewer_emb, noise_std)
        head_outputs = self.forward_heads(pooled)
        return {
            "logit": head_outputs["logits"] if "logits" in head_outputs else head_outputs["logit"],
            "symptom_logits": head_outputs["symptom_logits"],
            "diversity_loss": diversity_loss,
            "pooled_representation": pooled,
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

        # v16 Set Transformer
        instance_features = self.set_transformer(instance_features)

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
            "symptom_logits": torch.stack(batch_symptom_logits),
            "diversity_loss": torch.stack(batch_diversity_losses).mean(),
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


class SSDamilRClassifierV17(nn.Module):
    """v17: Multi-Sample Dropout (M-Drop) on v9d Backbone.

    Reverts complex v16 setups and introduces zero-parameter M-Drop
    in the classification heads to heavily regularize the dense layers
    during small dataset training.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        proj_dim: int = 64,
        num_symptoms: int = 8,
        dropout_rate: float = 0.3,
        temperature: float = 1.0,
        n_pool_heads: int = 2,
        num_mc_samples: int = 4,
    ):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.num_mc_samples = num_mc_samples
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

        self.pooling = MultiHeadAttentionPooling(
            input_dim=self.working_dim,
            hidden_dim=32,
            n_heads=n_pool_heads,
            temperature=temperature,
        )

        self.sym_dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(num_mc_samples)])
        self.symptom_head = nn.Sequential(
            nn.Linear(self.working_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_symptoms),
        )

        self.sym_projector = nn.Sequential(
            nn.Linear(num_symptoms, self.working_dim),
            nn.Tanh(),
        )
        self.inject_gate = nn.Linear(self.working_dim * 2, self.working_dim)
        nn.init.constant_(self.inject_gate.bias, 2.0)

        self.main_dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_mc_samples)])
        self.main_classifier = nn.Linear(self.working_dim, 1)

    def forward(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> dict:
        pooled, diversity_loss = self.forward_backbone(patient_emb, interviewer_emb, noise_std)
        head_outputs = self.forward_heads(pooled)
        return {
            "logit": head_outputs["logits"] if "logits" in head_outputs else head_outputs["logit"],
            "symptom_logits": head_outputs["symptom_logits"],
            "diversity_loss": diversity_loss,
            "pooled_representation": pooled,
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

        sym_logits_list = []
        for dropout in self.sym_dropouts:
            sym_logits_list.append(self.symptom_head(dropout(pooled)))
        sym_logits = torch.stack(sym_logits_list, dim=0).mean(dim=0)

        sym_signal = self.sym_projector(sym_logits)

        if is_batched:
            gate_input = torch.cat([pooled, sym_signal], dim=1)
        else:
            gate_input = torch.cat([pooled, sym_signal], dim=0).unsqueeze(0)

        gate = torch.sigmoid(self.inject_gate(gate_input))
        if not is_batched:
            gate = gate.squeeze(0)

        enriched = gate * pooled + (1.0 - gate) * sym_signal

        main_logits = []
        for dropout in self.main_dropouts:
            main_h = dropout(enriched)
            main_logits.append(self.main_classifier(main_h).squeeze(-1))
        main_logit = torch.stack(main_logits, dim=0).mean(dim=0)

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
            "symptom_logits": torch.stack(batch_symptom_logits),
            "diversity_loss": torch.stack(batch_diversity_losses).mean(),
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

class GatedMultiHeadAttentionPoolingV25(nn.Module):
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


class SSDamilRClassifierV25(nn.Module):
    """v25: Gated Attention Pooling + Attention Dropout + Gated Symptom Injection."""
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

        self.pooling = GatedMultiHeadAttentionPoolingV25(
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


class GLUProjector(nn.Module):
    """Gated Linear Unit (GLU) projector for more expressive feature selection."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        d = x.size(-1) // 2
        return x[..., :d] * torch.sigmoid(x[..., d:])


class SSDamilRClassifierV26(nn.Module):
    """v26: GLU Projector + Pre-LN Backbone + Manifold Mixup (V9 Core).

    Architectural differences from V9:
    - Uses a Gated Linear Unit (GLU) instead of Linear+ReLU for the projector,
      which allows for highly adaptive feature selection from the powerful PLM.
    - Uses Pre-LayerNorm (normalizing features BEFORE cross-attention and fusion)
      instead of Post-LayerNorm, improving gradient flow and training stability.
    - Retains Multi-Head Attention Pooling and Gated Symptom Injection.
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

        # 1. GLU Projector
        if proj_dim and proj_dim > 0:
            self.projector = GLUProjector(embedding_dim, proj_dim)
        else:
            self.projector = nn.Identity()

        # Pre-LN Normalizations
        self.norm_p = nn.LayerNorm(self.working_dim)
        self.norm_i = nn.LayerNorm(self.working_dim)

        # 2-3. Role Interaction (Backbone)
        self.cross_attention = CrossRoleAttention(dim=self.working_dim)
        self.fusion = RoleAwareFusion(dim=self.working_dim)

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
        pooled, _, diversity_loss = self.forward_backbone(patient_emb, interviewer_emb, noise_std)
        head_out = self.forward_heads(pooled)
        head_out["diversity_loss"] = diversity_loss
        head_out["pooled_representation"] = pooled
        # return mock head weights for backward compatibility
        head_out["head_weights"] = [] 
        return head_out

    def forward_backbone(
        self,
        patient_emb: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone (projector → Pre-LN → cross-attention → fusion → pooling)."""
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std
            interviewer_emb = interviewer_emb + torch.randn_like(interviewer_emb) * noise_std

        p_proj = self.projector(patient_emb)
        i_proj = self.projector(interviewer_emb)

        # Pre-LN
        p_norm = self.norm_p(p_proj)
        i_norm = self.norm_i(i_proj)

        context, _ = self.cross_attention(p_norm, i_norm)
        instance_features = self.fusion(p_norm, context)

        pooled, _, diversity_loss = self.pooling(instance_features)
        return pooled, diversity_loss

    def forward_heads(self, pooled: torch.Tensor) -> dict:
        """Run the classification heads on a (possibly mixed) pooled representation."""
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

            pooled, div_loss = self.forward_backbone(p_i, i_i, noise_std=noise_std)
            head_out = self.forward_heads(pooled)
            
            batch_main_logits.append(head_out["logit"])
            batch_symptom_logits.append(head_out["symptom_logits"])
            batch_diversity_losses.append(div_loss)

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


class SSDamilRClassifierV27(nn.Module):
    """v27: Early Linguistic Fusion + V9 Core.

    Fuses explicit clinical linguistic features (16-D) into the patient
    representations before the Cross-Role Attention, allowing the model
    to explicitly attend to utterances with clinical markers like
    past-tense rumination or disfluency.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_ling_features: int = 16,
        proj_dim: int = 64,
        num_symptoms: int = 8,
        dropout_rate: float = 0.3,
        temperature: float = 1.0,
        n_pool_heads: int = 2,
    ):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.working_dim = proj_dim if (proj_dim and proj_dim > 0) else embedding_dim

        # 1. Projector (Patient and Interviewer have different input dimensions)
        if proj_dim and proj_dim > 0:
            self.patient_projector = nn.Sequential(
                nn.Linear(embedding_dim + num_ling_features, proj_dim),
                nn.ReLU(),
            )
            self.interviewer_projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError("proj_dim must be > 0 for V27 Linguistic Fusion")

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
        patient_ling: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> dict:
        pooled, _, diversity_loss = self.forward_backbone(patient_emb, patient_ling, interviewer_emb, noise_std)
        head_out = self.forward_heads(pooled)
        head_out["diversity_loss"] = diversity_loss
        head_out["pooled_representation"] = pooled
        head_out["head_weights"] = []
        return head_out

    def forward_backbone(
        self,
        patient_emb: torch.Tensor,
        patient_ling: torch.Tensor,
        interviewer_emb: torch.Tensor,
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone with early linguistic fusion."""
        if self.training and noise_std > 0:
            patient_emb = patient_emb + torch.randn_like(patient_emb) * noise_std
            interviewer_emb = interviewer_emb + torch.randn_like(interviewer_emb) * noise_std
            # We don't add noise to linguistic features, as they are explicit statistics.

        p_concat = torch.cat([patient_emb, patient_ling], dim=-1)

        p_proj = self.patient_projector(p_concat)
        i_proj = self.interviewer_projector(interviewer_emb)

        context, _ = self.cross_attention(p_proj, i_proj)
        instance_features = self.post_fusion_norm(self.fusion(p_proj, context))

        pooled, _, diversity_loss = self.pooling(instance_features)
        return pooled, diversity_loss

    def forward_heads(self, pooled: torch.Tensor) -> dict:
        """Run the classification heads on a (possibly mixed) pooled representation."""
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

    def forward_batch(
        self,
        patient_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_ling_bags: torch.Tensor,
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
            p_l = patient_ling_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]

            pooled, div_loss = self.forward_backbone(p_i, p_l, i_i, noise_std=noise_std)
            head_out = self.forward_heads(pooled)
            
            batch_main_logits.append(head_out["logit"])
            batch_symptom_logits.append(head_out["symptom_logits"])
            batch_diversity_losses.append(div_loss)

        return {
            "logits": torch.stack(batch_main_logits),
            "symptom_logits": torch.stack(batch_symptom_logits),
            "diversity_loss": torch.stack(batch_diversity_losses).mean(),
        }

    def forward_batch_split(
        self,
        patient_bags: torch.Tensor,
        interviewer_bags: torch.Tensor,
        patient_ling_bags: torch.Tensor,
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
            p_l = patient_ling_bags[i, :p_n, :]
            i_i = interviewer_bags[i, :i_n, :]

            pooled, div_loss = self.forward_backbone(p_i, p_l, i_i, noise_std=noise_std)
            pooled_list.append(pooled)
            div_losses.append(div_loss)

        return torch.stack(pooled_list), torch.stack(div_losses).mean()

class SSDamilRClassifierV29(nn.Module):
    """v29: Temporal Convolutional MIL (TC-MIL).

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


class SSDamilRClassifierV30(SSDamilRClassifierV9):
    """v30: v9d Architecture + Multi-Sample Dropout (MSD).
    
    Retains the exact stable backbone of v9d, but implements an implicit 
    ensemble by passing the enriched representation through multiple independent 
    dropout masks. This provides extreme regularization for small datasets 
    without adding any parameters.
    """
    def __init__(self, n_msd_samples: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.n_msd_samples = n_msd_samples
        
        # We need independent dropout masks.
        dropout_rate = kwargs.get("dropout_rate", 0.3)
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(n_msd_samples)
        ])

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
        
        # Multi-Sample Dropout
        msd_logits = []
        for dropout in self.dropouts:
            main_h = dropout(enriched)
            msd_logits.append(self.main_classifier(main_h).squeeze(-1))
            
        msd_stacked = torch.stack(msd_logits, dim=0) # (n_samples, B) or (n_samples,)
        mean_logit = msd_stacked.mean(dim=0)
        
        key = "logits" if is_batched else "logit"
        return {
            key: mean_logit, 
            "symptom_logits": sym_logits,
            "msd_logits": msd_stacked
        }
