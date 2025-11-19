import cv2
import numpy as np
from sklearn.cluster import KMeans

class KeyframeDetector:
    def __init__(self, num_keyframes=5):
        self.num_keyframes = num_keyframes

    def extract_motion_vectors(self, video_path):
        """
        Approximation of motion analysis using optical flow magnitude.
        """
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        if not ret:
            return []
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_magnitudes = []
        frame_indices = []
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            avg_motion = np.mean(mag)
            motion_magnitudes.append(avg_motion)
            frame_indices.append(idx)
            
            prev_gray = gray
            idx += 1
            
        cap.release()
        return np.array(motion_magnitudes), np.array(frame_indices)

    def select_keyframes(self, video_path):
        motion_mags, indices = self.extract_motion_vectors(video_path)
        
        if len(motion_mags) < self.num_keyframes:
            return indices # Return all if video is short
            
        # Cluster frames based on motion intensity
        # High motion might indicate splicing or artifacts
        # We also want diversity, so we cluster 1D motion signal
        
        kmeans = KMeans(n_clusters=self.num_keyframes, n_init=10)
        labels = kmeans.fit_predict(motion_mags.reshape(-1, 1))
        
        selected_indices = []
        for i in range(self.num_keyframes):
            # Find frame closest to cluster center
            cluster_center = kmeans.cluster_centers_[i]
            cluster_indices = indices[labels == i]
            cluster_mags = motion_mags[labels == i]
            
            if len(cluster_mags) > 0:
                closest_idx = np.argmin(np.abs(cluster_mags - cluster_center))
                selected_indices.append(cluster_indices[closest_idx])
                
        return sorted(selected_indices)

if __name__ == "__main__":
    # Stub for testing
    print("KeyframeDetector initialized")
