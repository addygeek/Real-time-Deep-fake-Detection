import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    def __init__(self, input_dims, hidden_dim=256, num_classes=2):
        super(FusionTransformer, self).__init__()
        
        # input_dims: dict {'triage': 1, 'resilience': 512, 'av': 1}
        
        self.triage_proj = nn.Linear(input_dims['triage'], hidden_dim)
        self.resilience_proj = nn.Linear(input_dims['resilience'], hidden_dim)
        self.av_proj = nn.Linear(input_dims['av'], hidden_dim)
        
        # Learnable tokens for each modality
        self.modality_tokens = nn.Parameter(torch.randn(1, 3, hidden_dim))
        
        # Cross-Attention / Self-Attention
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=3
        )
        
        # Gating Mechanism (Learnable weights for each modality's reliability)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, triage_out, resilience_emb, av_score):
        batch_size = triage_out.size(0)
        
        # Project inputs to common dimension
        t_emb = self.triage_proj(triage_out)      # [Batch, Hidden]
        r_emb = self.resilience_proj(resilience_emb) # [Batch, Hidden]
        a_emb = self.av_proj(av_score)            # [Batch, Hidden]
        
        # Stack: [Batch, 3, Hidden]
        stacked = torch.stack([t_emb, r_emb, a_emb], dim=1)
        
        # Add modality tokens (optional, but helps transformer distinguish sources)
        stacked = stacked + self.modality_tokens
        
        # Transformer Fusion
        fused = self.transformer(stacked) # [Batch, 3, Hidden]
        
        # Flatten for gating calculation
        flat_fused = fused.view(batch_size, -1) # [Batch, 3*Hidden]
        
        # Calculate gate weights
        gate_weights = self.gate(flat_fused) # [Batch, 3]
        
        # Weighted sum of fused features
        # fused: [Batch, 3, Hidden] * gate_weights: [Batch, 3, 1]
        weighted_fused = torch.sum(fused * gate_weights.unsqueeze(-1), dim=1) # [Batch, Hidden]
        
        logits = self.classifier(weighted_fused)
        
        return logits, gate_weights
