# SpectraShield - Multimodal Deepfake Detection System

A comprehensive deepfake detection platform featuring advanced ML models, blockchain provenance, and real-time analysis.

## 🚀 Features

### Core Capabilities
- **Multi-Stage ML Pipeline**: Fast triage → Compression resilience → Audio-visual alignment → Multimodal fusion
- **Blockchain Provenance**: SHA-256 hashing with Merkle tree verification for tamper-proof audit trails
- **Real-time Updates**: WebSocket-based live status updates during analysis
- **Job Queue System**: BullMQ (Redis) for asynchronous video processing
- **RESTful API**: Complete backend API with Express.js

### ML Engine Components
1. **Fast Triage** (CNN-LSTM): 30-150ms inference for quick suspicious frame detection
2. **Compression Resilient Embeddings**: Two-stream CNN with denoising autoencoder
3. **Audio-Visual Alignment**: Wav2Vec2 + Lip landmark tracking + Sync Transformer
4. **Keyframe Localization**: Motion vector analysis with clustering
5. **Multimodal Fusion**: Cross-attention transformer with modality gating
6. **Adversarial Training**: GAN-based robustness enhancement
7. **Continual Learning**: Online model updates with replay buffer

## 📁 Project Structure

```
spectrashield/
├── backend/              # Node.js/Express API server
│   ├── api/
│   │   ├── controllers/  # Request handlers
│   │   ├── routes/       # API endpoints
│   │   ├── middlewares/  # Upload, validation
│   │   └── validators/   # Input validation
│   ├── core/
│   │   ├── blockchain/   # Provenance chain implementation
│   │   └── socket.js     # WebSocket server
│   ├── jobs/             # BullMQ queue definitions
│   ├── workers/          # Background job processors
│   ├── models/           # MongoDB schemas
│   └── config/           # DB, Redis configuration
├── ml-engine/            # Python ML pipeline
│   ├── cnn_lstm_fast_triage/
│   ├── compression_resilient_embeddings/
│   ├── audio_visual_alignment/
│   ├── keyframe_localization/
│   ├── multimodal_transformer_fusion/
│   ├── adversarial_generator/
│   ├── continual_learning/
│   ├── inference/        # Main pipeline orchestrator
│   └── api.py            # FastAPI wrapper
├── frontend/             # Next.js React application
│   ├── app/              # Pages and layouts
│   ├── components/       # React components
│   ├── services/         # API client
│   └── hooks/            # Custom React hooks
├── deployment/
│   ├── kubernetes/       # K8s manifests
│   └── terraform/        # Infrastructure as code
└── tests/                # Integration tests
```

## 🛠️ Installation

### Prerequisites
- Node.js 18+
- Python 3.9+
- MongoDB
- Redis
- (Optional) CUDA-capable GPU for ML inference

### Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd spectrashield

# Start all services
docker-compose up -d

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:4000
# ML Engine: http://localhost:5000
```

### Manual Setup

#### Backend
```bash
cd backend
npm install
cp ../.env.example .env
# Edit .env with your configuration
npm run dev
```

#### ML Engine
```bash
cd ml-engine
pip install -r requirements.txt
python api.py
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📡 API Endpoints

### Upload & Analysis
- `POST /upload` - Upload video for analysis
- `GET /analysis/status/:id` - Get analysis status
- `GET /analysis/results/:id` - Get analysis results

### Blockchain Provenance
- `POST /blockchain/verify` - Verify analysis on blockchain
- `POST /blockchain/compare` - Compare video hashes
- `GET /blockchain/stats` - Get blockchain statistics
- `GET /blockchain/block/:hash` - Get specific block

### Analytics
- `GET /analytics/summary` - System-wide statistics

### Model Management
- `POST /model/retrain` - Trigger adaptive learning

### System
- `GET /health` - Health check

## 🔧 Configuration

### Environment Variables

**Backend (.env)**
```env
PORT=4000
DB_URI=mongodb://localhost:27017/spectrashield
REDIS_HOST=localhost
REDIS_PORT=6379
ML_ENGINE_URL=http://localhost:5000
```

**Frontend**
```env
NEXT_PUBLIC_API_URL=http://localhost:4000
```

## 🧪 Testing

```bash
# Backend tests
cd backend
npm test

# Integration tests
cd tests
python integration_test.py

# ML Engine tests
cd ml-engine
python -m pytest
```

## 🚢 Deployment

### Kubernetes
```bash
cd deployment/kubernetes
kubectl apply -f deployments.yaml
kubectl apply -f services.yaml
```

### AWS (Terraform)
```bash
cd deployment/terraform
terraform init
terraform plan
terraform apply
```

## 🔐 Blockchain Architecture

The provenance layer uses a custom blockchain implementation:

- **Hashing**: SHA-256 for video fingerprinting
- **Consensus**: Proof-of-Work (configurable difficulty)
- **Verification**: Merkle tree-based integrity checks
- **Storage**: Persistent JSON-based chain storage
- **Features**:
  - Tamper-proof audit trails
  - Video hash comparison
  - Reupload mismatch detection
  - Chain validation

## 📊 ML Pipeline Workflow

```
Video Upload
    ↓
[Fast Triage] → Suspicion score
    ↓
[Keyframe Extraction] → Select critical frames
    ↓
[Parallel Processing]
    ├─ [Compression Resilience] → Robust embeddings
    └─ [AV Alignment] → Sync score
    ↓
[Multimodal Fusion] → Final verdict
    ↓
[Blockchain Recording] → Provenance hash
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- PyTorch team for deep learning framework
- Hugging Face for Transformers library
- OpenCV community for computer vision tools
- MongoDB and Redis teams for database solutions

## 📧 Support

For issues and questions, please open a GitHub issue or contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: November 2025