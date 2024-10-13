import cv2
import numpy as np
from tqdm import tqdm
import sys
import os

CONFIG = {
    'dot_radius': 2,
    'black_color': (0, 0, 0),
}

def read_positions_file(file_path):
    frame_data = {}
    with open(file_path, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 5:
                frame, _, x, y, _ = parts
                frame = int(frame)
                x, y = float(x), float(y)
                if frame not in frame_data:
                    frame_data[frame] = []
                frame_data[frame].append((x, y))
    return frame_data

def draw_points(frame, points):
    for x, y in points:
        cv2.circle(frame, (int(x), int(y)), CONFIG['dot_radius'], CONFIG['black_color'], -1)
    return frame

def process_video(video_path, positions_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_data = read_positions_file(positions_path)
    max_frame = max(frame_data.keys())

    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        start_frame = i * 10 + 1
        end_frame = min(start_frame + 9, max_frame)
        
        points = []
        for j in range(start_frame, end_frame + 1):
            if j in frame_data:
                points.extend(frame_data[j])

        annotated_frame = draw_points(frame, points)
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a video file path as an argument.")
        sys.exit(1)

    video_path = sys.argv[1]
    folder_path = os.path.dirname(video_path)
    positions_path = os.path.join(folder_path, "positions_warped_grouped.txt")
    output_path = os.path.join(folder_path, "wormy.mp4")

    process_video(video_path, positions_path, output_path)
    print(f"Video saved as: {output_path}")