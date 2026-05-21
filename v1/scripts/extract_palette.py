# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python-headless",
#     "scikit-learn",
#     "numpy",
# ]
# ///

import cv2
import numpy as np
from sklearn.cluster import KMeans
import json

def extract_colors(video_path, num_colors=15):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Sample every 30th frame
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 30 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (100, 100)) # Resize for speed
            frames.append(frame)
        frame_idx += 1
        
    cap.release()
    
    if not frames:
        print("No frames extracted.")
        return []
        
    pixels = np.vstack(frames).reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Enhance contrast and saturation
    colors_uint8 = np.uint8(colors).reshape(1, -1, 3)
    hsv = cv2.cvtColor(colors_uint8, cv2.COLOR_RGB2HSV).astype(float)
    
    # Boost saturation by 50%
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
    
    # Boost contrast by stretching Value channel around 128
    v = hsv[..., 2]
    v = 128 + (v - 128) * 1.5
    hsv[..., 2] = np.clip(v, 0, 255)
    
    hsv = np.uint8(hsv)
    enhanced_colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    
    hex_colors = ["#{:02x}{:02x}{:02x}".format(c[0], c[1], c[2]) for c in enhanced_colors]
    return hex_colors

if __name__ == "__main__":
    video_path = "chinatown-palette-1.mp4"
    print(f"Extracting colors from {video_path}...")
    palette = extract_colors(video_path, 15)
    print("Extracted Palette:")
    print(palette)
    with open("palette.json", "w") as f:
        json.dump(palette, f)
