import cv2
import numpy as np
import sys
import os
from tqdm import tqdm

def create_mean_image(input_video):
    # Open the video file
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate total seconds
    total_seconds = total_frames // fps
    
    # List to store one frame per second
    frames = []
    
    with tqdm(total=total_seconds, desc="Processing video") as pbar:
        for second in range(total_seconds):
            # Set the position to the start of each second
            cap.set(cv2.CAP_PROP_POS_FRAMES, second * fps)
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to float32 for averaging
            frames.append(frame.astype(np.float32))
            
            pbar.update(1)
    
    # Release the video capture object
    cap.release()
    
    # Calculate the mean image
    if frames:
        mean_image = np.mean(frames, axis=0).astype(np.uint8)
        
        # Save the mean image
        output_path = "median_aura.png"
        cv2.imwrite(output_path, mean_image)
        print(f"Mean image saved as {output_path}")
    else:
        print("No frames were processed. The video might be empty or corrupted.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_video_path>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    if not os.path.exists(input_video):
        print(f"Error: The file {input_video} does not exist.")
        sys.exit(1)
    
    create_mean_image(input_video)