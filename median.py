import cv2
import numpy as np
import sys
import os

def extract_median_frame(video_path, num_samples=20):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame indices to sample
    sample_indices = np.linspace(0, frame_count - 1, num_samples, dtype=int)
    
    # Initialize array to store sampled frames
    sampled_frames = np.zeros((num_samples, frame_height, frame_width, 3), dtype=np.uint8)
    
    for i, frame_index in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            sampled_frames[i] = frame
        else:
            print(f"Warning: Could not read frame at index {frame_index}")
    
    # Release the video capture object
    cap.release()
    
    # Calculate the median frame
    median_frame = np.median(sampled_frames, axis=0).astype(np.uint8)
    
    return median_frame

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bg.py <video_file_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: The file {video_path} does not exist.")
        sys.exit(1)
    
    # Extract the median frame
    median_frame = extract_median_frame(video_path)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}_median.png"
    
    # Save the median frame
    cv2.imwrite(output_filename, median_frame)
    print(f"Median frame saved as {output_filename}")

if __name__ == "__main__":
    main()