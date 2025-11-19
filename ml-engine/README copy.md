# SpectraShield ML Engine

Python-based machine learning pipeline for deepfake detection.

## Modules

### 1. Fast Triage (`cnn_lstm_fast_triage/`)
Lightweight CNN-LSTM model for rapid initial screening (30-150ms inference).

### 2. Compression Resilient Embeddings (`compression_resilient_embeddings/`)
- Two-stream CNN (RGB + Noise residuals)
- Denoising autoencoder
- Social media compression simulation

### 3. Audio-Visual Alignment (`audio_visual_alignment/`)
- Wav2Vec2 for phoneme extraction
- Lip landmark tracking (dlib)
- Transformer-based sync analysis

### 4. Keyframe Localization (`keyframe_localization/`)
- Optical flow motion analysis
- KMeans clustering for frame selection
- Manipulation hotspot detection

### 5. Multimodal Fusion (`multimodal_transformer_fusion/`)
- Cross-attention mechanism
- Modality gating for reliability weighting
- Final classification

### 6. Adversarial Generator (`adversarial_generator/`)
- GAN-based hard negative generation
- Robustness training loop

### 7. Continual Learning (`continual_learning/`)
- Online model updates
- Replay buffer for catastrophic forgetting prevention
- Meta-learning adaptability

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Standalone Inference
```bash
python inference/pipeline.py path/to/video.mp4
```

### API Server
```bash
python api.py
```

The API will be available at `http://localhost:5000`

### Endpoints
- `POST /predict` - Run inference on a video
- `GET /health` - Health check

## Training

Models are currently initialized with pre-trained backbones or random weights. To train:

1. Prepare dataset (e.g., FaceForensics++, DFDC)
2. Implement training scripts in `training/` directory
3. Run training for each module
4. Save weights to `models/` directory

## Requirements

- PyTorch 2.0+
- Transformers 4.30+
- OpenCV 4.8+
- Dlib 19.24+
- CUDA (optional, for GPU acceleration)

## Architecture

```
Video Input
    ↓
[Preprocessing & Frame Extraction]
    ↓
[Fast Triage] → Early exit if confidence is very high/low
    ↓
[Keyframe Selection] → Select top-k suspicious frames
    ↓
[Parallel Processing]
    ├─ [Compression Resilience]
    └─ [Audio-Visual Alignment]
    ↓
[Multimodal Fusion]
    ↓
Final Result (JSON)
```

## Output Format

```json
{
  "filename": "video.mp4",
  "is_fake": true,
  "confidence": 0.87,
  "artifacts": {
    "audio_mismatch": 0.65,
    "visual_anomalies": 0.82
  },
  "gate_weights": [[0.3, 0.5, 0.2]]
}
```
