import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

import face_recognition

def extract_frames_from_video(video_path, output_dir, num_frames=15):
    """
    Extracts a fixed number of evenly spaced frames from a video and saves them as JPEGs.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If the video is too short, we just take what we can
    if total_frames <= 0:
        return
        
    # Calculate evenly spaced indices
    # e.g., if total_frames=100, num_frames=10 -> [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    saved_count = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Save frame
            out_path = os.path.join(output_dir, f"{base_filename}_frame{idx:04d}.jpg")
            locations = face_recognition.face_locations(frame)
            if len(locations) > 0:
                top, rt, bot, lt = locations[0]
                try:
                    cv2.imwrite(out_path, frame[top:bot, lt:rt])
                    saved_count += 1
                except:
                    continue
            
    cap.release()

def process_dataset(input_pattern, output_base, label_dir, num_frames=15):
    """
    Finds all videos matching the input_pattern and processes them.
    """
    videos = glob(input_pattern, recursive=True)
    out_dir = os.path.join(output_base, label_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Found {len(videos)} videos for {label_dir}. Extracting {num_frames} frames per video...")
    for video in tqdm(videos, desc=f"Processing {label_dir}"):
        extract_frames_from_video(video, out_dir, num_frames)

if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Output directory for extracted frames
    OUTPUT_DIR = os.path.join(DATA_DIR, "extracted_frames")
    
    # Process Real Videos (Originals)
    real_pattern = os.path.join(DATA_DIR, "original_sequences", "**", "*.mp4")
    process_dataset(real_pattern, OUTPUT_DIR, "real", num_frames=15)
    
    # Process Fake Videos (Deepfakes, FaceSwap, Face2Face)
    fake_pattern = os.path.join(DATA_DIR, "manipulated_sequences", "**", "*.mp4")
    process_dataset(fake_pattern, OUTPUT_DIR, "fake", num_frames=15)
    
    print("\nExtraction complete! Now your Jupyter Notebook can load fast .jpg images.")
