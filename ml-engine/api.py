"""
SpectraShield ML Engine - Production API
Uses actual ML models for deepfake detection
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from detector import LightweightDeepfakeDetector
    DETECTOR_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load detector: {e}")
    DETECTOR_AVAILABLE = False

app = FastAPI(title="SpectraShield ML Engine", version="2.4.0")

# Initialize detector
detector = None
if DETECTOR_AVAILABLE:
    try:
        detector = LightweightDeepfakeDetector()
        print("✓ ML Detector initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize detector: {e}")

class VideoRequest(BaseModel):
    filePath: str

@app.post("/predict")
async def predict(request: VideoRequest):
    """
    Run deepfake detection on video
    """
    start_time = time.time()
    
    # Check if file exists
    video_path = Path(request.filePath)
    if not video_path.exists():
        # Try relative path from uploads
        video_path = Path(__file__).parent.parent / "backend" / "uploads" / video_path.name
        
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {request.filePath}")
    
    try:
        if detector and DETECTOR_AVAILABLE:
            # Use actual detector
            result = detector.analyze_video(str(video_path))
        else:
            # Fallback to mock
            result = _mock_prediction(video_path.name)
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = int(processing_time)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def _mock_prediction(filename):
    """Fallback mock prediction"""
    import random
    
    is_fake = random.random() > 0.5
    confidence = random.uniform(0.75, 0.95) if is_fake else random.uniform(0.05, 0.25)
    
    return {
        "filename": filename,
        "is_fake": is_fake,
        "confidence": confidence,
        "artifacts": {
            "visual_anomalies": random.uniform(0.1, 0.9),
            "temporal_inconsistency": random.uniform(0.0, 0.5)
        },
        "model_version": "v2.4.0-mock",
        "mode": "mock"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "detector_available": DETECTOR_AVAILABLE,
        "detector_loaded": detector is not None,
        "version": "2.4.0",
        "mode": "production" if detector else "fallback"
    }

@app.get("/")
def root():
    return {
        "service": "SpectraShield ML Engine",
        "version": "2.4.0",
        "status": "operational",
        "detector": "active" if detector else "mock",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("SpectraShield ML Engine - Production Mode")
    print("=" * 70)
    print(f"Detector Status: {'✓ Active' if detector else '⚠ Fallback Mode'}")
    print("Running on: http://localhost:5000")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
