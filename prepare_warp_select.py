import cv2
import numpy as np
import os
import sys

def load_config(config_path):
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                config[key.strip()] = value.strip()
    return config

def save_config(config_path, config):
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    with open(config_path, 'w') as f:
        for line in lines:
            if '=' in line:
                key = line.split('=')[0].strip()
                if key in config:
                    f.write(f"{key} = {config[key]}\n")
                else:
                    f.write(line)
            else:
                f.write(line)
        
        for key, value in config.items():
            if not any(key in line for line in lines):
                f.write(f"{key} = {value}\n")

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

def click_event(event, x, y, flags, param):
    global points, image, window_name

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            points.append([x, y])
            cv2.imshow(window_name, image)

def main():
    global points, image, window_name

    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    median_path = os.path.join(folder_path, "median.png")
    config_path = os.path.join(folder_path, "config.txt")

    if not os.path.exists(median_path):
        print(f"Error: The file {median_path} does not exist.")
        sys.exit(1)

    config = load_config(config_path)

    window_name = "Select Points"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        points = []
        image = cv2.imread(median_path)
        original_image = image.copy()

        cv2.putText(image, "Select 4 points for homography", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, image)
        cv2.setMouseCallback(window_name, click_event)

        while len(points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                sys.exit(0)

        src_points = np.array(points, dtype=np.float32)
        cropped_image, adjusted_points = crop_lower_percentage(original_image, src_points, config)
        result, matrix = apply_homography(cropped_image, adjusted_points, config)

        cv2.imshow("Result", result[0])

        # Calculate and save warped corner positions
        height, width = cropped_image.shape[:2]
        corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        
        min_x, min_y = np.min(warped_corners, axis=0)
        warped_corners -= [min_x, min_y]
        x_offset = (int(config['output_width']) - (np.max(warped_corners[:, 0]) - np.min(warped_corners[:, 0]))) / 2 + int(config['x_offset'])
        y_offset = int(config['output_height']) - (np.max(warped_corners[:, 1]) - np.min(warped_corners[:, 1])) - int(config['y_offset'])
        warped_corners += [x_offset, y_offset]

        # Save points and warped corners to config
        config['topleft'] = f"{points[0][0]},{points[0][1]}"
        config['topright'] = f"{points[1][0]},{points[1][1]}"
        config['bottomright'] = f"{points[2][0]},{points[2][1]}"
        config['bottomleft'] = f"{points[3][0]},{points[3][1]}"

        corner_names = ['warped_topleft', 'warped_topright', 'warped_bottomright', 'warped_bottomleft']
        for name, corner in zip(corner_names, warped_corners):
            config[name] = f"{int(round(corner[0]))},{int(round(corner[1]))}"

        save_config(config_path, config)

        print("Points and warped corners saved to config file. Press 'f' to try again or ESC to exit.")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('f'):
                break
            elif key == 27:  # ESC key
                cv2.destroyAllWindows()
                sys.exit(0)

        if key == ord('f'):
            cv2.destroyWindow("Result")
            continue
        else:
            break

if __name__ == "__main__":
    main()