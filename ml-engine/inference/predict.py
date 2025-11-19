import sys
import json
import random
import time

def analyze_video(video_path):
    """
    Mock inference function for deepfake detection.
    In a real scenario, this would load the model and process the video frames.
    """
    # Simulate processing time
    time.sleep(2)
    
    # Mock results
    is_fake = random.choice([True, False])
    confidence = random.uniform(0.85, 0.99) if is_fake else random.uniform(0.01, 0.15)
    
    result = {
        "filename": video_path,
        "is_fake": is_fake,
        "confidence": confidence,
        "artifacts": {
            "visual_inconsistencies": random.uniform(0, 1),
            "audio_mismatch": random.uniform(0, 1)
        },
        "timestamp": time.time()
    }
    
    return result

if __name__ == "__main__":
    # Simple CLI interface
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(json.dumps(analyze_video(video_path), indent=2))
    else:
        print(json.dumps({"error": "No video path provided"}))
