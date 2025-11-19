import torch
import torch.nn as nn
import torchvision.models as models

class FastTriageNet(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1):
        super(FastTriageNet, self).__init__()
        
        # Lightweight backbone: MobileNetV3 Small
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Remove classification head, keep feature extractor
        # MobileNetV3 Small feature size is 576 before classifier
        self.backbone.classifier = nn.Identity()
        self.feature_dim = 576 
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, C, H, W)
        Returns:
            suspicion_score: Tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # Merge batch and sequence dimensions for CNN processing
        c_in = x.view(batch_size * seq_len, C, H, W)
        
        # Extract features
        features = self.backbone(c_in)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Use the output of the last time step
        last_hidden = lstm_out[:, -1, :]
        
        # Classification
        suspicion_score = self.classifier(last_hidden)
        
        return suspicion_score

if __name__ == "__main__":
    # Quick test
    model = FastTriageNet()
    dummy_input = torch.randn(2, 10, 3, 224, 224) # Batch=2, Seq=10
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [2, 1]
