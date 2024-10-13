import cv2
import numpy as np
from tqdm import tqdm
import sys
import os

CONFIG = {
    'fade_rate': 0.999,
    'new_overlay_intensity': 0.4,
    'overlay_blend_factor': 0.8,
    'dot_radius': 0,
    'interaction_radius': 16,
    'min_distance': 5,
    'green_color': (0, 0, 0),
    'heatmap_fade_rate': 0.999,
    'heatmap_growth_rate': 0.001,
    'mask_fade_rate': 0.9995,
    'mask_intensity': 0.9,
    'heatmap_max_value': 1.0,
    'draw_green_dots': True,
}

class CircleAnnotator:
    def __init__(self, background_shape):
        self.heatmap = np.zeros(background_shape[:2], dtype=np.float32)
        self.mask = np.zeros(background_shape[:2], dtype=np.float32)
        self.background_shape = background_shape
        self.interaction_mask = np.zeros((CONFIG['interaction_radius']*2+1, CONFIG['interaction_radius']*2+1), dtype=np.float32)
        cv2.circle(self.interaction_mask, (CONFIG['interaction_radius'], CONFIG['interaction_radius']), CONFIG['interaction_radius'], 1, thickness=-1)

    def annotate(self, background, positions):
        frame = background.copy()
        new_overlay = np.zeros(self.background_shape[:2], dtype=np.float32)
        
        positions = np.array(positions)
        x, y, group = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Vectorized distance calculation
        distances = np.sqrt(((x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2))
        
        valid_interactions = (CONFIG['min_distance'] < distances) & (distances <= 2 * CONFIG['interaction_radius'])
        valid_interactions &= (group[:, np.newaxis] != group) | ((group[:, np.newaxis] == -1) & (group == -1))
        np.fill_diagonal(valid_interactions, False)
        
        interaction_indices = np.argwhere(valid_interactions)
        
        for i, j in interaction_indices:
            x1, y1 = int(x[i]), int(y[i])
            x2, y2 = int(x[j]), int(y[j])
            
            # Calculate the overlap region
            left = max(x1 - CONFIG['interaction_radius'], x2 - CONFIG['interaction_radius'], 0)
            right = min(x1 + CONFIG['interaction_radius'], x2 + CONFIG['interaction_radius'], self.background_shape[1])
            top = max(y1 - CONFIG['interaction_radius'], y2 - CONFIG['interaction_radius'], 0)
            bottom = min(y1 + CONFIG['interaction_radius'], y2 + CONFIG['interaction_radius'], self.background_shape[0])
            
            if left < right and top < bottom:
                mask1 = self.interaction_mask[
                    top - (y1 - CONFIG['interaction_radius']):bottom - (y1 - CONFIG['interaction_radius']),
                    left - (x1 - CONFIG['interaction_radius']):right - (x1 - CONFIG['interaction_radius'])
                ]
                mask2 = self.interaction_mask[
                    top - (y2 - CONFIG['interaction_radius']):bottom - (y2 - CONFIG['interaction_radius']),
                    left - (x2 - CONFIG['interaction_radius']):right - (x2 - CONFIG['interaction_radius'])
                ]
                
                overlap = cv2.multiply(mask1, mask2)
                new_overlay[top:bottom, left:right] += overlap

        # Draw green dots
        if CONFIG['draw_green_dots']:
            for xi, yi in zip(x.astype(int), y.astype(int)):
                cv2.circle(frame, (xi, yi), CONFIG['dot_radius'], CONFIG['green_color'], -1)

        # Update heatmap and mask
        self.heatmap = cv2.addWeighted(self.heatmap, CONFIG['heatmap_fade_rate'], 
                                       new_overlay, CONFIG['heatmap_growth_rate'], 0)
        self.heatmap = np.clip(self.heatmap, 0, CONFIG['heatmap_max_value'])

        self.mask = cv2.addWeighted(self.mask, CONFIG['mask_fade_rate'], 
                                    new_overlay, CONFIG['mask_intensity'], 0)
        self.mask = np.clip(self.mask, 0, 1)

        # Create color heatmap
        normalized_heatmap = (self.heatmap / CONFIG['heatmap_max_value'] * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_PLASMA)

        # Apply mask to heatmap
        mask_3channel = cv2.cvtColor((self.mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        masked_heatmap = cv2.multiply(heatmap_color, mask_3channel, scale=1./255)

        # Blend heatmap on top of the frame
        frame = cv2.addWeighted(frame, 1, masked_heatmap, CONFIG['overlay_blend_factor'], 0)

        return frame


def read_positions_file(file_path):
    frame_data = {}
    with open(file_path, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 5:
                frame, _, x, y, group = parts
                frame = int(frame)
                x, y = float(x), float(y)
                group = int(group)
                if frame not in frame_data:
                    frame_data[frame] = []
                frame_data[frame].append((x, y, group))
    return frame_data

def read_config(config_path):
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                config[key.strip()] = value.strip()
    return config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a folder path as an argument.")
        sys.exit(1)

    folder_path = sys.argv[1]
    positions_path = os.path.join(folder_path, "positions_warped_grouped.txt")
    background_path = os.path.join(folder_path, "median_warped.png")
    output_path = os.path.join(folder_path, "annotated_video.mp4")
    config_path = os.path.join(folder_path, "config.txt")

    # Read the config file
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    file_config = read_config(config_path)
    
    # Get FPS from config, default to 30 if not found
    fps = float(file_config.get('FPS', 30))
    print(f"Using FPS: {fps}")

    background = cv2.imread(background_path)
    if background is None:
        print(f"Error: Could not read background image from {background_path}")
        sys.exit(1)

    height, width = background.shape[:2]

    # Read the positions file
    frame_data = read_positions_file(positions_path)

    circle_annotator = CircleAnnotator(background.shape)

    # Set up the video writer with the correct FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    out.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'avc1'))
    out.set(cv2.CAP_PROP_BITRATE, 20000)  # Set bitrate to 20k

    for frame_num in tqdm(sorted(frame_data.keys())):
        positions = frame_data[frame_num]
        annotated_frame = circle_annotator.annotate(background, positions)
        out.write(annotated_frame)

    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved as: {output_path}")