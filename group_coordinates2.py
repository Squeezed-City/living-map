import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import Pool, cpu_count

def read_proximity_threshold(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('proximity_threshold'):
                key, value = line.split('=')
                return float(value.strip())
    raise ValueError("proximity_threshold not found in config file")

class OptimizedAdaptiveGroupAnalyzer:
    def __init__(self, proximity_threshold, window_size=600, min_samples=2):
        self.proximity_threshold = proximity_threshold
        self.window_size = window_size
        self.min_samples = min_samples
        self.tracks = {}
        self.groups = {}
        self.next_group_id = 0

    def update_positions(self, frame_num, positions):
        for track_id, pos in positions.items():
            if track_id not in self.tracks:
                self.tracks[track_id] = []
            self.tracks[track_id].append((frame_num, pos[0], pos[1]))

    def analyze_groups(self):
        all_frames = sorted(set(frame for track in self.tracks.values() for frame, _, _ in track))
        max_frame = max(all_frames)

        with tqdm(total=(max_frame - self.window_size) // (self.window_size // 2) + 1, desc="Analyzing frames") as pbar:
            with Pool(processes=cpu_count()) as pool:
                for start_frame in range(0, max_frame, self.window_size // 2):
                    end_frame = min(start_frame + self.window_size, max_frame)
                    
                    active_tracks = [
                        (track_id, np.array([(f, x, y) for f, x, y in frames if start_frame <= f < end_frame]))
                        for track_id, frames in self.tracks.items()
                        if any(start_frame <= f < end_frame for f, _, _ in frames)
                    ]
                    
                    if len(active_tracks) < 2:
                        pbar.update(1)
                        continue

                    track_ids, positions = zip(*active_tracks)
                    mean_positions = np.array([np.mean(pos[:, 1:], axis=0) for pos in positions])
                    
                    clustering = MiniBatchKMeans(n_clusters=min(len(mean_positions) // 2, 10), random_state=0).fit(mean_positions)
                    
                    tree = cKDTree(mean_positions)
                    pairs = list(tree.query_pairs(r=self.proximity_threshold))
                    
                    if pairs:
                        results = pool.starmap(self.check_pair, [(positions[i], positions[j]) for i, j in pairs])
                        for (i, j), should_merge in zip(pairs, results):
                            if should_merge:
                                self.merge_groups(track_ids[i], track_ids[j])
                    
                    pbar.update(1)

    def check_pair(self, pos1, pos2):
        common_frames = np.intersect1d(pos1[:, 0], pos2[:, 0])
        if len(common_frames) == 0:
            return False
        
        idx1 = np.searchsorted(pos1[:, 0], common_frames)
        idx2 = np.searchsorted(pos2[:, 0], common_frames)
        
        distances = np.linalg.norm(pos1[idx1, 1:] - pos2[idx2, 1:], axis=1)
        return np.mean(distances) <= self.proximity_threshold

    def merge_groups(self, track_id1, track_id2):
        group1 = self.groups.get(track_id1)
        group2 = self.groups.get(track_id2)

        if group1 is None and group2 is None:
            new_group = self.next_group_id
            self.next_group_id += 1
            self.groups[track_id1] = new_group
            self.groups[track_id2] = new_group
        elif group1 is None:
            self.groups[track_id1] = group2
        elif group2 is None:
            self.groups[track_id2] = group1
        elif group1 != group2:
            for track_id, group in self.groups.items():
                if group == group2:
                    self.groups[track_id] = group1

    def get_group_id(self, track_id):
        return self.groups.get(track_id, -1)

def analyze_groups(folder_path, segment_duration=120):  # 120 seconds = 2 minutes
    config_path = os.path.join(folder_path, "config.txt")
    proximity_threshold = read_proximity_threshold(config_path)

    input_path = os.path.join(folder_path, "positions_warped.txt")
    output_path = os.path.join(folder_path, "positions_warped_grouped.txt")

    df = pd.read_csv(input_path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    frame_data = {int(frame): {int(row['Track_ID']): (float(row['X']), float(row['Y'])) 
                               for _, row in group.iterrows()} 
                  for frame, group in df.groupby('Frame')}

    all_frames = sorted(frame_data.keys())
    max_frame = max(all_frames)
    fps = 30  # Adjust this based on your video's frame rate

    analyzer = OptimizedAdaptiveGroupAnalyzer(proximity_threshold=proximity_threshold)

    for start_frame in range(0, max_frame, segment_duration * fps):
        end_frame = min(start_frame + segment_duration * fps, max_frame)
        segment_frames = range(start_frame, end_frame)
        
        segment_analyzer = OptimizedAdaptiveGroupAnalyzer(proximity_threshold=proximity_threshold)
        
        for frame in segment_frames:
            if frame in frame_data:
                segment_analyzer.update_positions(frame, frame_data[frame])
        
        segment_analyzer.analyze_groups()
        
        for track_id, group_id in segment_analyzer.groups.items():
            analyzer.groups[track_id] = group_id + analyzer.next_group_id
        
        analyzer.next_group_id += segment_analyzer.next_group_id

    with open(output_path, 'w') as f:
        f.write("Frame,Track_ID,X,Y,Group_ID\n")
        for frame in sorted(frame_data.keys()):
            for track_id, position in frame_data[frame].items():
                group_id = analyzer.get_group_id(track_id)
                f.write(f"{frame},{track_id},{position[0]:.3f},{position[1]:.3f},{group_id}\n")

    print(f"Group analysis complete. Results written to: {output_path}")
    print(f"Number of groups: {analyzer.next_group_id}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the folder path as an argument.")
        sys.exit(1)

    folder_path = sys.argv[1]
    analyze_groups(folder_path)