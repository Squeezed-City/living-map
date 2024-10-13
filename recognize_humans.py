import cv2
import numpy as np
from tqdm import tqdm
import os
from ultralytics import YOLO


# this takes the original video, recognizes people and saves the coordinates of their feet to file
# input: video_original.mp4
# output: positions_original.txt

def recognize_people(video_path, output_path):
    model = YOLO('yolov8s-visdrone-enot.pt')
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_labels = ['pedestrian', 'person']

    tracks = {}
    next_id = 0
    distance_threshold = 50

    with open(output_path, 'w') as f:
        f.write("Frame, Track_ID, X, Y\n")

        for frame_num in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=1280)
            
            detections = []
            for r in results:
                for box in r.boxes:
                    label = model.names[int(box.cls)]
                    if label.lower() in target_labels:
                        x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                        foot_x = (x1 + x2) / 2
                        foot_y = y2
                        detections.append((foot_x, foot_y))

            new_tracks = {}
            unmatched_detections = list(range(len(detections)))

            for track_id, track in tracks.items():
                if unmatched_detections:
                    distances = [np.linalg.norm(np.array(track['position']) - np.array(detections[i])) 
                                 for i in unmatched_detections]
                    closest_detection_index = np.argmin(distances)
                    min_distance = distances[closest_detection_index]

                    if min_distance <= distance_threshold:
                        detection_index = unmatched_detections[closest_detection_index]
                        track['position'] = detections[detection_index]
                        new_tracks[track_id] = track
                        f.write(f"{frame_num}, {track_id}, {track['position'][0]:.3f}, {track['position'][1]:.3f}\n")
                        unmatched_detections.pop(closest_detection_index)

            for detection in [detections[i] for i in unmatched_detections]:
                tracks[next_id] = {'position': detection}
                new_tracks[next_id] = tracks[next_id]
                f.write(f"{frame_num}, {next_id}, {detection[0]:.3f}, {detection[1]:.3f}\n")
                next_id += 1

            tracks = new_tracks

    cap.release()
    print(f"Recognition complete. Results written to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Please provide a folder path as an argument.")
        sys.exit(1)

    folder_path = sys.argv[1]
    video_path = os.path.join(folder_path, "video_original.mp4")
    output_path = os.path.join(folder_path, "positions_original.txt")

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        sys.exit(1)

    recognize_people(video_path, output_path)