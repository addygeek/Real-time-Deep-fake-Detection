import sys
import os
import json
import torch
import numpy as np

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn_lstm_fast_triage.model import FastTriageNet
from compression_resilient_embeddings.model import ResilientEmbedder
from audio_visual_alignment.model import AVSyncModel
from multimodal_transformer_fusion.model import FusionTransformer
from keyframe_localization.detector import KeyframeDetector

class DeepfakeDetectionPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading models on {self.device}...")
        
        # Initialize Models
        self.triage_model = FastTriageNet().to(self.device)
        self.resilience_model = ResilientEmbedder().to(self.device)
        self.av_model = AVSyncModel().to(self.device)
        self.fusion_model = FusionTransformer(
            input_dims={'triage': 1, 'resilience': 512, 'av': 1}
        ).to(self.device)
        
        self.keyframe_detector = KeyframeDetector(num_keyframes=5)
        
        # Load weights (Mocking this part as we don't have trained weights)
        self.triage_model.eval()
        self.resilience_model.eval()
        self.av_model.eval()
        self.fusion_model.eval()

    def predict(self, video_path):
        if not os.path.exists(video_path):
            return {"error": "File not found"}

        try:
            # 1. Keyframe Extraction
            # In a real pipeline, we might run Triage on all frames first.
            # Here we extract keyframes for heavy lifting.
            keyframes_indices = self.keyframe_detector.select_keyframes(video_path)
            
            # Mocking feature extraction results for demonstration
            # Real implementation would load frames, preprocess, and pass through models
            
            # Triage Score (Mock)
            triage_score = torch.tensor([[0.8]]).to(self.device) # Suspicious
            
            # Resilience Embedding (Mock)
            resilience_emb = torch.randn(1, 512).to(self.device)
            
            # AV Sync Score (Mock)
            av_score = torch.tensor([[0.2]]).to(self.device) # Low sync -> Suspicious
            
            # Fusion
            with torch.no_grad():
                logits, gate_weights = self.fusion_model(triage_score, resilience_emb, av_score)
                probs = torch.softmax(logits, dim=1)
                fake_prob = probs[0, 1].item()
                
            result = {
                "filename": os.path.basename(video_path),
                "is_fake": fake_prob > 0.5,
                "confidence": fake_prob,
                "artifacts": {
                    "audio_mismatch": 1.0 - av_score.item(),
                    "visual_anomalies": triage_score.item()
                },
                "gate_weights": gate_weights.tolist()
            }
            
            return result

        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    
    pipeline = DeepfakeDetectionPipeline()
    result = pipeline.predict(args.video_path)
    print(json.dumps(result, indent=2))
