import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AVSyncModel(nn.Module):
    def __init__(self, hidden_size=256):
        super(AVSyncModel, self).__init__()
        
        # Audio Encoder (Pre-trained Wav2Vec2)
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.feature_extractor._freeze_parameters() # Freeze feature extractor
        self.audio_proj = nn.Linear(768, hidden_size)
        
        # Visual Encoder (Lip Landmarks Sequence Encoder)
        # Input: (Batch, Seq, 68*2) -> 68 landmarks * 2 coords
        self.visual_encoder = nn.LSTM(
            input_size=136, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True
        )
        self.visual_proj = nn.Linear(hidden_size * 2, hidden_size) # *2 for bidirectional
        
        # Fusion Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4),
            num_layers=2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_input, visual_input):
        """
        Args:
            audio_input: (Batch, Audio_Seq) raw waveform
            visual_input: (Batch, Video_Seq, 136) landmarks
        """
        # Audio Features
        audio_out = self.audio_encoder(audio_input).last_hidden_state # [Batch, Audio_Seq, 768]
        audio_emb = self.audio_proj(audio_out) # [Batch, Audio_Seq, Hidden]
        
        # Visual Features
        visual_out, _ = self.visual_encoder(visual_input) # [Batch, Video_Seq, Hidden*2]
        visual_emb = self.visual_proj(visual_out) # [Batch, Video_Seq, Hidden]
        
        # Align sequences (Simple truncation/padding or Cross-Attention needed here)
        # For simplicity, we assume pre-aligned or use pooling
        
        # Global Average Pooling for this stub
        audio_pool = torch.mean(audio_emb, dim=1)
        visual_pool = torch.mean(visual_emb, dim=1)
        
        # Simple similarity check (Cosine Similarity could be used directly)
        # Here we fuse and classify
        combined = (audio_pool + visual_pool) / 2
        
        score = self.classifier(combined)
        
        return score
