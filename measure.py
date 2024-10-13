import cv2
import numpy as np
import sys
import os

def extract_median_frame(video_path, num_samples=50):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_indices = np.linspace(0, frame_count - 1, num_samples, dtype=int)
    sampled_frames = np.zeros((num_samples, frame_height, frame_width, 3), dtype=np.uint8)

    for i, frame_index in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            sampled_frames[i] = frame
        else:
            print(f"Warning: Could not read frame at index {frame_index}")

    cap.release()
    median_frame = np.median(sampled_frames, axis=0).astype(np.uint8)
    return frame_height, frame_width, median_frame

def save_config(config_path, data):
    existing_lines = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing_lines = f.readlines()

    updated_lines = []
    keys_written = set()
    for line in existing_lines:
        line = line.strip()
        if line.startswith('#') or not line:
            updated_lines.append(line + '\n')
        elif '=' in line:
            key = line.split('=')[0].strip()
            if key in data:
                updated_lines.append(f"{key} = {data[key]}\n")
                keys_written.add(key)
            else:
                updated_lines.append(line + '\n')

    for key, value in data.items():
        if key not in keys_written:
            updated_lines.append(f"{key} = {value}\n")

    with open(config_path, 'w') as f:
        f.writelines(updated_lines)

def load_config(config_path):
    data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = [x.strip() for x in line.split('=', 1)]
                        data[key] = value
                    except ValueError:
                        print(f"Warning: Skipping invalid line in config file: {line}")
    return data

def click_event(event, x, y, flags, param):
    global points, image, window_name, config, config_path

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            points.append((x, y))
            cv2.imshow(window_name, image)

            if len(points) == 2:
                distance = np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
                proximity_threshold = round(distance)
                config['proximity_threshold'] = str(proximity_threshold)
                save_config(config_path, config)
                cv2.destroyAllWindows()
                print(f"Proximity threshold set to {proximity_threshold}. Configuration saved. Exiting script.")
                os._exit(0)  # Force exit the script

def main():
    global points, image, window_name, config, config_path

    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    video_path = os.path.join(folder_path, "video_birdsview.mp4")
    median_path = os.path.join(folder_path, "median_warped.png")
    config_path = os.path.join(folder_path, "config.txt")

    if not os.path.exists(video_path):
        print(f"Error: The file {video_path} does not exist.")
        sys.exit(1)

    # Extract and save median frame
    _, _, median_frame = extract_median_frame(video_path)
    cv2.imwrite(median_path, median_frame)
    print(f"Median frame saved as {median_path}")

    # Load the first frame for user interaction
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read the first frame of the video.")
        sys.exit(1)

    # Load existing config or create new one
    config = load_config(config_path)

    # Use the first frame for point selection
    image = first_frame

    window_name = "Select Points"
    points = []

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    image_with_text = image.copy()
    cv2.putText(image_with_text, "Click on two points that measure a meter", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(window_name, image_with_text)

    cv2.setMouseCallback(window_name, click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("Script terminated by user.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()