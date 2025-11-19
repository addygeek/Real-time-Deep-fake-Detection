import cv2
import numpy as np

def create_dummy_video(filename="dummy.mp4", duration=2, fps=30):
    height, width = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        # Create a frame with moving rectangle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = int((i / (duration * fps)) * width)
        cv2.rectangle(frame, (x, 50), (x+50, 100), (0, 255, 0), -1)
        out.write(frame)
        
    out.release()
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_video()
