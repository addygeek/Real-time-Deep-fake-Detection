from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import random
from datetime import datetime

app = FastAPI(title="SpectraShield ML Engine - Demo Mode")

class VideoRequest(BaseModel):
    filePath: str

@app.post("/predict")
async def predict(request: VideoRequest):
    """
    Demo prediction endpoint - returns mock results
    In production, this would run the full ML pipeline
    """
    
    # Simulate processing time
    import time
    time.sleep(1)
    
    # Generate realistic mock results
    is_fake = random.random() > 0.5
    confidence = random.uniform(0.75, 0.95) if is_fake else random.uniform(0.05, 0.25)
    
    result = {
        "filename": request.filePath.split("/")[-1] if "/" in request.filePath else request.filePath.split("\\")[-1],
        "is_fake": is_fake,
        "confidence": confidence,
        "artifacts": {
            "audio_mismatch": random.uniform(0.1, 0.9) if is_fake else random.uniform(0.0, 0.2),
            "visual_anomalies": random.uniform(0.6, 0.95) if is_fake else random.uniform(0.0, 0.3),
            "compression_artifacts": random.uniform(0.4, 0.8),
            "temporal_inconsistency": random.uniform(0.3, 0.7) if is_fake else random.uniform(0.0, 0.2)
        },
        "gate_weights": [[0.3, 0.5, 0.2]],
        "processing_time_ms": random.randint(800, 2500),
        "model_version": "v2.4.0-demo",
        "timestamp": datetime.now().isoformat()
    }
    
    return result

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "mode": "demo",
        "model_loaded": True,
        "version": "2.4.0",
        "message": "ML Engine running in demo mode (mock predictions)"
    }

@app.get("/")
def root():
    return {
        "service": "SpectraShield ML Engine",
        "version": "2.4.0",
        "mode": "demo",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("SpectraShield ML Engine - Demo Mode")
    print("=" * 60)
    print("Running on: http://localhost:5000")
    print("Mode: DEMO (Mock predictions)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=5000)
