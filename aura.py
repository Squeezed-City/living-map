import cv2
import numpy as np
from tqdm import tqdm
import sys
import os
import subprocess

def read_config(config_path):
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                config[key.strip()] = value.strip()
    return config

def speed_up_video(input_path, output_path, fps):
    ffmpeg_cmd = f"ffmpeg -i {input_path} -filter:v \"setpts=0.1*PTS\" -r {fps} -an {output_path}"
    subprocess.call(ffmpeg_cmd, shell=True)

def process_video(folder_path):
    # Read config file
    config_path = os.path.join(folder_path, "config.txt")
    config = read_config(config_path)
    fps = float(config.get('FPS', 30))
    
    # Calculate sliding window size
    window_size = round(fps * 2)
    
    print(f"Using FPS: {fps}")
    print(f"Sliding window size: {window_size}")

    # Generate input and output paths
    input_path = os.path.join(folder_path, "annotated_video.mp4")
    speed_up_output = os.path.join(folder_path, "annotated_video_10x.mp4")
    final_output = os.path.join(folder_path, "annotated_video_aura.mp4")

    # Step 1: Speed up video using FFmpeg
    print("Speeding up video...")
    speed_up_video(input_path, speed_up_output, fps)

    # Step 2: Process the sped-up video
    print("Applying sliding window effect...")
    cap = cv2.VideoCapture(speed_up_output)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object for the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(final_output, fourcc, fps, (width, height))
    
    # Buffer to store frames for sliding window
    frame_buffer = []
    
    # Process the video
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_buffer.append(frame)
            
            if len(frame_buffer) == window_size:
                window_mean = np.mean(frame_buffer, axis=0).astype(np.uint8)
                out.write(window_mean)
                frame_buffer.pop(0)
            
            pbar.update(1)
    
    # Process remaining frames in buffer
    while frame_buffer:
        window_mean = np.mean(frame_buffer, axis=0).astype(np.uint8)
        out.write(window_mean)
        frame_buffer.pop(0)
    
    # Release resources
    cap.release()
    out.release()
    
    # Convert to high quality H.264 MP4
    print("Converting to high-quality H.264...")
    convert_to_h264(final_output, os.path.join(folder_path, "annotated_video_aura_final.mp4"), fps)
    
    # Remove intermediate files
    os.remove(speed_up_output)
    os.remove(final_output)

def convert_to_h264(input_path, output_path, fps):
    ffmpeg_cmd = f"ffmpeg -i {input_path} -c:v libx264 -preset slow -crf 17 -r {fps} -c:a copy {output_path}"
    subprocess.call(ffmpeg_cmd, shell=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.exists(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        sys.exit(1)

    process_video(folder_path)
    print(f"Processing complete. Output saved as {os.path.join(folder_path, 'annotated_video_aura_final.mp4')}")