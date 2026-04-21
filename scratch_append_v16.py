import sys

with open("models/ss_damil_r.py", "a") as f:
    f.write('''

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
''')
