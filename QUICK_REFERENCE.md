# SpectraShield - Quick Reference Guide

## 🚀 Quick Start

### Start All Services
```powershell
# Windows
.\start-all.bat

# Linux/Mac
./start-all.sh
```

### Access Points
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:4000
- **ML Engine**: http://localhost:5000

---

## 📁 Project Structure

```
spectrashield/
├── backend/          # Node.js API (Port 4000)
├── frontend/         # Next.js UI (Port 3000)
├── ml-engine/        # Python ML (Port 5000)
├── deployment/       # Docker, K8s, Terraform
├── tests/            # Integration tests
└── docs/             # Documentation
```

---

## 🔧 Common Commands

### Backend
```powershell
cd backend
npm install          # Install dependencies
npm run dev          # Start dev server
npm test             # Run tests
```

### Frontend
```powershell
cd frontend
npm install          # Install dependencies
npm run dev          # Start dev server
npm run build        # Build for production
```

### ML Engine
```powershell
cd ml-engine
pip install -r requirements-minimal.txt  # Install
python train.py      # Train model
python api.py        # Start API server
```

---

## 📡 API Quick Reference

### Upload Video
```bash
curl -X POST http://localhost:4000/upload \
  -F "video=@test.mp4"
```

### Check Status
```bash
curl http://localhost:4000/analysis/status/ANALYSIS_ID
```

### Get Results
```bash
curl http://localhost:4000/analysis/results/ANALYSIS_ID
```

### Verify Blockchain
```bash
curl -X POST http://localhost:4000/blockchain/verify \
  -H "Content-Type: application/json" \
  -d '{"analysisId":"ANALYSIS_ID"}'
```

---

## 🐛 Troubleshooting

### Port Already in Use
```powershell
# Find process
netstat -ano | findstr :4000

# Kill process
taskkill /PID <PID> /F
```

### Services Not Starting
```powershell
# Check status
.\check-status.bat

# Test system
.\test-system.bat
```

### ML Engine Issues
```powershell
# Reinstall dependencies
pip install --upgrade -r requirements-minimal.txt

# Use demo mode
python api-demo.py
```

---

## 📊 File Locations

### Uploaded Videos
```
backend/uploads/
```

### Blockchain Data
```
backend/data/blockchain.json
```

### ML Model Weights
```
ml-engine/models/frame_classifier.pth
```

### Logs
- Backend: Console output
- Frontend: Browser console
- ML Engine: Console output

---

## 🔑 Environment Variables

### Backend (.env)
```env
PORT=4000
DB_URI=mongodb://localhost:27017/spectrashield
REDIS_HOST=localhost
REDIS_PORT=6379
ML_ENGINE_URL=http://localhost:5000
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:4000
```

---

## 🧪 Testing

### Quick Test
```powershell
.\test-system.bat
```

### Integration Test
```powershell
cd tests
python integration_test.py
```

### Health Checks
```bash
curl http://localhost:4000/health
curl http://localhost:5000/health
```

---

## 📦 Dependencies

### Backend
- Express, Socket.io, BullMQ
- Mongoose, Redis, Multer

### Frontend
- Next.js, React, TypeScript
- Tailwind CSS

### ML Engine
- PyTorch, OpenCV, FastAPI
- NumPy, Uvicorn

---

## 🎯 Key Features

✅ Video upload & analysis
✅ Real-time status updates
✅ ML-based detection
✅ Blockchain verification
✅ Analytics dashboard
✅ Processing queue

---

## 📞 Support

### Documentation
- `README.md` - Overview
- `TECH.md` - Technical details
- `COMPLETE_GUIDE.md` - Full guide
- `TESTING.md` - Testing guide

### Scripts
- `start-all.bat` - Start everything
- `check-status.bat` - Check services
- `test-system.bat` - Run tests
- `complete-setup.bat` - Full setup

---

## ⚡ Performance Tips

1. **Use GPU** for ML inference
2. **Enable caching** in Redis
3. **Scale horizontally** with Docker
4. **Optimize database** indexes
5. **Use CDN** for frontend

---

## 🎓 Learning Resources

### Code Examples
- Backend: `backend/api/controllers/`
- Frontend: `frontend/components/`
- ML: `ml-engine/detector.py`

### Architecture
- See `TECH.md` for diagrams
- Check `deployment/` for configs

---

## ✅ Checklist

### Before Deployment
- [ ] All tests passing
- [ ] Environment variables set
- [ ] Database configured
- [ ] Redis running
- [ ] ML model trained
- [ ] Blockchain initialized

### Production Ready
- [ ] Docker images built
- [ ] Kubernetes manifests ready
- [ ] Monitoring configured
- [ ] Backups automated
- [ ] SSL certificates installed

---

**Quick Links:**
- [Main README](README.md)
- [Technical Docs](TECH.md)
- [Complete Guide](COMPLETE_GUIDE.md)
- [Deployment Guide](deployment/DEPLOYMENT.md)

**Status**: ✅ All systems operational
**Version**: 2.4.0
