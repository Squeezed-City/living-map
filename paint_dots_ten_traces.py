import cv2
import numpy as np
from tqdm import tqdm
import sys
import os

CONFIG = {
    'dot_radius': 3,
    'trace_thickness': 2,  # New configuration for trace thickness
    'black_color': (0, 0, 0),
    'green_color': (0, 255, 0),
    'fade_rate': 0.9,
    'trace_intensity': 0.3,
    'num_trace_frames': 10
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

def draw_points(frame, points, color, radius):
    for x, y in points:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    return frame

def draw_traces(frame, points, color, thickness):
    for x, y in points:
        cv2.circle(frame, (int(x), int(y)), thickness, color, -1)
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

    trace_overlay = np.zeros((frame_height, frame_width, 3), dtype=np.float32)

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

        # Fade the existing trace
        trace_overlay *= CONFIG['fade_rate']

        # Add new points to the trace
        temp_overlay = np.zeros_like(trace_overlay)
        draw_traces(temp_overlay, points, CONFIG['green_color'], CONFIG['trace_thickness'])
        trace_overlay = cv2.addWeighted(trace_overlay, 1, temp_overlay, CONFIG['trace_intensity'], 0)

        # Combine the original frame with the trace overlay
        combined_frame = cv2.addWeighted(frame, 1, trace_overlay.astype(np.uint8), 1, 0)

        # Draw the current points in black
        annotated_frame = draw_points(combined_frame, points, CONFIG['black_color'], CONFIG['dot_radius'])

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
    output_path = os.path.join(folder_path, "wormy_traces.mp4")

    process_video(video_path, positions_path, output_path)
    print(f"Video saved as: {output_path}")