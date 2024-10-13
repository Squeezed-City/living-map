import numpy as np
import sys
import os
from tqdm import tqdm
import cv2

def load_config(config_file):
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                config[key.strip()] = value.strip()
    return config

def load_warped_corners_from_config(config):
    corner_keys = ['warped_topleft', 'warped_topright', 'warped_bottomright', 'warped_bottomleft']
    corners = []
    for key in corner_keys:
        if key not in config:
            raise ValueError(f"Error: '{key}' not found in config file.")
        corners.append(list(map(float, config[key].split(','))))
    return np.array(corners, dtype=np.float32)

def adjust_points_for_crop(points, original_height, crop_height):
    adjusted_points = points.copy()
    adjusted_points[:, 1] -= (original_height - crop_height)
    return adjusted_points

def apply_homography_to_coordinates(input_file, output_file, corners, config):
    original_height, original_width = map(int, config['ORIGINAL_IMAGE_SIZE'].split(','))
    crop_height = int(original_height * float(config['crop_height_percentage']))
    cropped_image_size = (crop_height, original_width)

    src_points = np.array([[0, 0],
                           [original_width - 1, 0],
                           [original_width - 1, crop_height - 1],
                           [0, crop_height - 1]], dtype=np.float32)

    dst_points = corners

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        header = fin.readline()
        fout.write(header)

        for line in tqdm(fin, desc="Warping coordinates"):
            data = line.strip().split(',')
            frame, track_id, x, y = map(float, data[:4])

            y -= (original_height - crop_height)

            if y < 0:
                continue

            point = np.array([[[x, y]]], dtype=np.float32)
            warped_point = cv2.perspectiveTransform(point, matrix)[0][0]
            
            warped_x, warped_y = warped_point

            output_width = int(config['output_width'])
            output_height = int(config['output_height'])
            warped_x = max(0, min(warped_x, output_width - 1))
            warped_y = max(0, min(warped_y, output_height - 1))

            fout.write(f"{int(frame)}, {int(track_id)}, {warped_x:.3f}, {warped_y:.3f}\n")
    
    print(f"Warped coordinates saved to '{output_file}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python warp_coordinates.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    input_file = os.path.join(folder_path, "positions_original.txt")
    config_file = os.path.join(folder_path, "config.txt")
    output_file = os.path.join(folder_path, "positions_warped.txt")

    try:
        config = load_config(config_file)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)

    try:
        corners = load_warped_corners_from_config(config)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    if len(corners) != 4:
        print(f"Error: Expected 4 corner points in config, but found {len(corners)}.")
        sys.exit(1)

    apply_homography_to_coordinates(input_file, output_file, corners, config)