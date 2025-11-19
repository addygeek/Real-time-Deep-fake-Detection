# SpectraShield Testing Guide

## Test Suite Overview

### Backend Tests
- Unit tests for controllers
- Integration tests for API endpoints
- Blockchain verification tests
- Database operations tests

### Frontend Tests
- Component tests
- API integration tests
- E2E tests

### ML Engine Tests
- Model inference tests
- Pipeline integration tests
- Performance benchmarks

## Running Tests

### Backend

```bash
cd backend

# Install test dependencies
npm install --save-dev jest supertest

# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- controllers/blockchain.test.js
```

### Frontend

```bash
cd frontend

# Install test dependencies
npm install --save-dev @testing-library/react @testing-library/jest-dom

# Run tests
npm test

# Run E2E tests
npm run test:e2e
```

### ML Engine

```bash
cd ml-engine

# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_pipeline.py
```

## Integration Testing

### Full System Test

```bash
# 1. Start all services
docker-compose up -d

# 2. Wait for services to be ready
sleep 10

# 3. Run integration tests
cd tests
python integration_test.py
```

### Expected Output

```
Running Integration Tests...
✅ Health Check Passed
Testing Upload...
✅ Upload Passed. Analysis ID: 507f1f77bcf86cd799439011
ℹ️ Initial Status: queued
```

## Manual Testing Checklist

### Video Upload Flow
- [ ] Upload valid MP4 file
- [ ] Upload invalid file type (should fail)
- [ ] Upload file > 50MB (should fail)
- [ ] Drag and drop upload
- [ ] Progress bar displays correctly

### Analysis Flow
- [ ] Status changes from queued → processing → completed
- [ ] Real-time updates via WebSocket
- [ ] Results display correctly
- [ ] Confidence scores are reasonable (0-100)

### Blockchain Verification
- [ ] Verify button works
- [ ] Blockchain hash is generated
- [ ] Verification status shows as verified
- [ ] Block index is displayed
- [ ] Timestamp is correct

### Analytics Dashboard
- [ ] Total count displays
- [ ] Fake/Real distribution shows
- [ ] Recent analyses list populates
- [ ] Charts render correctly

### Error Handling
- [ ] Network errors show user-friendly messages
- [ ] Invalid IDs return 404
- [ ] Server errors return 500
- [ ] Validation errors return 400

## Performance Testing

### Load Testing with Apache Bench

```bash
# Test upload endpoint
ab -n 100 -c 10 -p video.mp4 -T multipart/form-data \
  http://localhost:4000/upload

# Test status endpoint
ab -n 1000 -c 50 \
  http://localhost:4000/analysis/status/507f1f77bcf86cd799439011
```

### Expected Performance
- Upload endpoint: < 2s response time
- Status endpoint: < 100ms response time
- Results endpoint: < 200ms response time
- ML inference: 2-10s depending on video length

## Security Testing

### Input Validation

```bash
# Test SQL injection (should be prevented)
curl -X POST http://localhost:4000/blockchain/verify \
  -H "Content-Type: application/json" \
  -d '{"analysisId":"'; DROP TABLE analyses;--"}'

# Test XSS (should be sanitized)
curl -X POST http://localhost:4000/upload \
  -F "video=<script>alert('xss')</script>"
```

### File Upload Security

```bash
# Test malicious file upload
curl -X POST http://localhost:4000/upload \
  -F "video=@malicious.exe"
# Expected: 400 Bad Request

# Test oversized file
curl -X POST http://localhost:4000/upload \
  -F "video=@large_file.mp4"
# Expected: 400 Bad Request if > 50MB
```

## Blockchain Testing

### Verify Chain Integrity

```bash
# Get chain stats
curl http://localhost:4000/blockchain/stats

# Expected response:
{
  "success": true,
  "stats": {
    "length": 5,
    "isValid": true,
    "latestBlock": "abc123...",
    "totalAnalyses": 4
  }
}
```

### Test Video Comparison

```bash
curl -X POST http://localhost:4000/blockchain/compare \
  -H "Content-Type: application/json" \
  -d '{
    "originalHash": "abc123...",
    "newVideoPath": "new_video.mp4"
  }'

# Expected: Mismatch score between 0-100
```

## ML Engine Testing

### Test Inference Pipeline

```bash
# Create test video
cd ml-engine
python create_dummy_video.py

# Run inference
python inference/pipeline.py dummy.mp4

# Expected output:
{
  "filename": "dummy.mp4",
  "is_fake": true,
  "confidence": 0.87,
  "artifacts": {
    "audio_mismatch": 0.65,
    "visual_anomalies": 0.82
  }
}
```

### Test Individual Modules

```python
# Test Fast Triage
from cnn_lstm_fast_triage.model import FastTriageNet
import torch

model = FastTriageNet()
dummy_input = torch.randn(1, 10, 3, 224, 224)
output = model(dummy_input)
assert output.shape == (1, 1)

# Test Fusion
from multimodal_transformer_fusion.model import FusionTransformer

model = FusionTransformer({'triage': 1, 'resilience': 512, 'av': 1})
triage = torch.randn(1, 1)
resilience = torch.randn(1, 512)
av = torch.randn(1, 1)
logits, gates = model(triage, resilience, av)
assert logits.shape == (1, 2)
```

## Database Testing

### MongoDB

```bash
# Connect to MongoDB
mongosh mongodb://localhost:27017/spectrashield

# Check collections
show collections

# Count analyses
db.analyses.count()

# Find recent analyses
db.analyses.find().sort({createdAt: -1}).limit(5)
```

### Redis

```bash
# Connect to Redis
redis-cli

# Check queue
LLEN bull:video-processing:wait

# Check active jobs
LLEN bull:video-processing:active

# Monitor in real-time
MONITOR
```

## Monitoring & Debugging

### Check Logs

```bash
# Backend logs
tail -f backend/logs/app.log

# Docker logs
docker-compose logs -f backend
docker-compose logs -f ml-engine
```

### Health Checks

```bash
# Backend health
curl http://localhost:4000/health

# ML Engine health
curl http://localhost:5000/health

# MongoDB health
mongosh --eval "db.adminCommand('ping')"

# Redis health
redis-cli ping
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Push to main/develop branches
- Pull requests

View results at: `https://github.com/your-repo/actions`

### Local CI Simulation

```bash
# Run all tests like CI would
./scripts/ci-test.sh
```

## Test Data

### Sample Videos

Create test videos with different characteristics:

```python
# Real video (no manipulation)
# Fake video (deepfake)
# Low quality (compressed)
# High quality (uncompressed)
# Short (< 5s)
# Long (> 30s)
```

### Test Cases

1. **Happy Path**: Valid video → Successful analysis → Blockchain verification
2. **Error Path**: Invalid file → 400 error
3. **Edge Case**: Very short video (1 frame)
4. **Edge Case**: Very long video (5 minutes)
5. **Concurrent**: Multiple uploads simultaneously

## Regression Testing

After each deployment:

```bash
# Run full test suite
npm run test:all

# Check critical paths
./scripts/smoke-test.sh

# Verify blockchain integrity
curl http://localhost:4000/blockchain/stats
```

## Test Coverage Goals

- Backend: > 80%
- Frontend: > 70%
- ML Engine: > 60%

## Reporting Issues

When reporting bugs, include:
1. Steps to reproduce
2. Expected behavior
3. Actual behavior
4. Error logs
5. Environment (OS, Node version, etc.)

## Test Automation

### Pre-commit Hooks

```bash
# Install husky
npm install --save-dev husky

# Add pre-commit hook
npx husky add .husky/pre-commit "npm test"
```

### Scheduled Tests

Run nightly tests for:
- Performance regression
- Memory leaks
- Database integrity
- Blockchain validation

---

**Remember**: Good tests are the foundation of reliable software. Test early, test often! 🧪
