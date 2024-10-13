import cv2
import numpy as np
import sys
import os
from tqdm import tqdm


def load_config(config_file):
    config = {}
    with open(config_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                config[key.strip()] = value.strip()
    return config, lines

def save_config(config_file, config, original_lines):
    with open(config_file, 'w') as f:
        for line in original_lines:
            line = line.strip()
            if line and not line.startswith('#'):
                key = line.split('=')[0].strip()
                if key in config:
                    f.write(f"{key} = {config[key]}\n")
                    del config[key]
            else:
                f.write(line + '\n')
        
        # Write any new keys that weren't in the original file
        for key, value in config.items():
            f.write(f"{key} = {value}\n")

def load_points_from_config(config):
    return np.array([
        [float(x) for x in config['topleft'].split(',')],
        [float(x) for x in config['topright'].split(',')],
        [float(x) for x in config['bottomright'].split(',')],
        [float(x) for x in config['bottomleft'].split(',')]
    ], dtype=np.float32)

def calculate_bounds(M, width, height):
    corners = np.array([[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]])
    transformed_corners = np.dot(M, corners.T).T
    transformed_corners /= transformed_corners[:, 2][:, np.newaxis]
    return np.min(transformed_corners[:, 0]), np.min(transformed_corners[:, 1]), np.max(transformed_corners[:, 0]), np.max(transformed_corners[:, 1])

def custom_warp_perspective(img, M, output_size, config):
    height, width = img.shape[:2]
    min_x, min_y, max_x, max_y = calculate_bounds(M, width, height)
    warped_width, warped_height = max_x - min_x, max_y - min_y

    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
    adjusted_M = np.dot(translation_matrix, M)

    warped_img = cv2.warpPerspective(img, adjusted_M, (int(np.ceil(warped_width)), int(np.ceil(warped_height))), borderValue=(255, 255, 255))

    final_img = np.full((output_size[1], output_size[0], 3), 255, dtype=np.uint8)

    x_offset = (output_size[0] - warped_img.shape[1]) / 2 + int(config['x_offset'])
    y_offset = output_size[1] - warped_img.shape[0] - int(config['y_offset'])

    src_rect = [max(0, -x_offset), max(0, -y_offset), 
                min(warped_img.shape[1], output_size[0]-x_offset) - max(0, -x_offset),
                min(warped_img.shape[0], output_size[1]-y_offset) - max(0, -y_offset)]
    
    dst_rect = [max(0, x_offset), max(0, y_offset),
                src_rect[2], src_rect[3]]

    src_rect = list(map(int, src_rect))
    dst_rect = list(map(int, dst_rect))

    final_img[dst_rect[1]:dst_rect[1]+dst_rect[3], dst_rect[0]:dst_rect[0]+dst_rect[2]] = \
        warped_img[src_rect[1]:src_rect[1]+src_rect[3], src_rect[0]:src_rect[0]+src_rect[2]]

    return final_img, x_offset, y_offset, warped_width, warped_height

def apply_homography(image, src_points, config):
    dst_points = np.array([[0, 0],
                           [int(config['dst_width']) - 1, 0],
                           [int(config['dst_width']) - 1, int(config['dst_height']) - 1],
                           [0, int(config['dst_height']) - 1]], dtype=np.float32)
    matrix, _ = cv2.findHomography(src_points, dst_points)
    return custom_warp_perspective(image, matrix, (int(config['output_width']), int(config['output_height'])), config), matrix

def crop_lower_percentage(image, points, config):
    height, width = image.shape[:2]
    crop_height = int(height * float(config['crop_height_percentage']))
    cropped_image = image[height - crop_height:, :]
    adjusted_points = points.copy()
    adjusted_points[:, 1] -= (height - crop_height)
    return cropped_image, adjusted_points

def process_video(video_path, config, output_folder):
    src_points = load_points_from_config(config)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Unable to open video '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = os.path.join(output_folder, "video_birdsview.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(config['output_width']), int(config['output_height'])))
    out.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'avc1'))
    out.set(cv2.CAP_PROP_BITRATE, 15000)

    ret, frame = cap.read()
    if not ret:
        raise IOError("Error: Unable to read the first frame of the video")

    cropped_frame, adjusted_points = crop_lower_percentage(frame, src_points, config)
    result, matrix = apply_homography(cropped_frame, adjusted_points, config)
    
    # Calculate and save warped corner positions
    height, width = cropped_frame.shape[:2]
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    
    min_x, min_y = np.min(warped_corners, axis=0)
    warped_corners -= [min_x, min_y]
    x_offset = (int(config['output_width']) - (np.max(warped_corners[:, 0]) - np.min(warped_corners[:, 0]))) / 2 + int(config['x_offset'])
    y_offset = int(config['output_height']) - (np.max(warped_corners[:, 1]) - np.min(warped_corners[:, 1])) - int(config['y_offset'])
    warped_corners += [x_offset, y_offset]

    # Save warped corners to config
    corner_names = ['warped_topleft', 'warped_topright', 'warped_bottomright', 'warped_bottomleft']
    for name, corner in zip(corner_names, warped_corners):
        config[name] = f"{int(round(corner[0]))},{int(round(corner[1]))}"

    # Process video frames with progress bar
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cropped_frame, adjusted_points = crop_lower_percentage(frame, src_points, config)
            result, _ = apply_homography(cropped_frame, adjusted_points, config)
            out.write(result[0])
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Bird's-eye view video saved as '{output_path}'")
    print("Warped corners saved to config file")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python apply_homography.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    video_path = os.path.join(folder_path, "video_original.mp4")
    config_path = os.path.join(folder_path, "config.txt")

    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)

    try:
        config, original_lines = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)

    print("Starting video processing...")
    try:
        process_video(video_path, config, folder_path)
        save_config(config_path, config, original_lines)  # Save updated config with warped corners
        print("Video processing completed successfully.")
    except IOError as e:
        print(str(e))
        sys.exit(1)


