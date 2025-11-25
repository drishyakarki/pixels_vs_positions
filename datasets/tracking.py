import json
import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm


def _init_process_worker(seed):
    """
    initializes random seeds for worker processes during parallel data loading.
    ensures reproducibility across multiprocessing workers.
    """
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class PlayerPositionDataset(Dataset):
    """
    dataset for loading player tracking data for group activity recognition.
    
    loads tracking data from parquet files and extracts temporal windows around annotated events.
    each sample consists of a sequence of graphs where nodes represent players/ball and edges
    encode spatial relationships based on the specified edge connectivity scheme.
    
    features per node (8 dimensions):
        0: x coordinate (pitch length direction)
        1: y coordinate (pitch width direction)
        2: is_ball flag (one-hot)
        3: home_team flag (one-hot)
        4: away_team flag (one-hot)
        5: delta_x (displacement from previous frame)
        6: delta_y (displacement from previous frame)
        7: z coordinate (ball height, -200 for players)
    
    normalization bounds derived from data analysis:
        pitch half-length (x): 85m (max observed: 80.04m)
        pitch half-width (y): 50m (max observed: 48.31m)
        max displacement: 110 (max observed: 101)
        max ball height (z): 30m (max observed: 25m)
    """
    
    def __init__(self, data_dir, split='train', window_size=16, frame_interval=9,
                 normalize=True, k=8, transforms=None, num_workers=None, 
                 time_tolerance_ms=10, edge=None, random_positioning=True, 
                 max_position_shift=4, random_positioning_prob=0.5, seed=42):
        
        self.data_dir = data_dir
        self.split = split
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.normalize = normalize
        self.k = k
        self.transforms = transforms
        self.time_tolerance_ms = time_tolerance_ms
        self.edge = edge
        self.num_objects = 23  # 11 home + 11 away + 1 ball
        self.feature_dim = 8
        self.pitch_half_length = 85
        self.pitch_half_width = 50
        self.max_displacement = 110
        self.max_ball_height = 30.0
        self.num_workers = num_workers
        self.random_positioning = random_positioning
        self.max_position_shift = max_position_shift
        self.random_positioning_prob = random_positioning_prob
        self.seed = seed

        self._load_all_data()

    def _load_all_data(self):
        """
        loads all tracking data into memory by processing parquet files in parallel.
        builds edge indices for each frame during loading to avoid recomputation during training.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        with open(os.path.join(self.data_dir, self.split, f"{self.split}.json"), 'r') as f:
            split_data = json.load(f)

        # prepare tasks for parallel processing with deterministic seeding
        tasks = [
            (os.path.basename(video['path']).replace('.parquet', ''),
             os.path.join(self.data_dir, self.split, video['path']),
             video.get('annotations'),
             self.seed,
             idx) 
            for idx, video in enumerate(split_data['videos'])
        ]

        all_samples = []
        
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_init_process_worker,
            initargs=(self.seed,)
        ) as executor:
            futures = [executor.submit(self._process_video_with_seed, *task) for task in tasks]
            for future in tqdm(futures, total=len(tasks), desc=f"Loading {self.split} videos"):
                samples, _, _, _, _ = future.result()
                all_samples.extend(samples)

        # build label mappings
        label_counts = defaultdict(int)
        for sample in all_samples:
            label_counts[sample['label']] += 1

        self.labels = sorted(label_counts.keys())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # convert samples to final format
        self.processed_samples = []
        for sample in tqdm(all_samples, desc="Processing samples"):
            self.processed_samples.append({
                'features': sample['features'],
                'edge_indices': sample['edge_indices'],
                'label': self.label_to_idx[sample['label']],
                'label_name': sample['label'],
                'window_id': sample['window_id'],
                'game_time': sample['game_time'],
                'team': sample['team'],
                'match_id': sample['match_id']
            })
        
        print(f"\nLabel distribution in {self.split} set:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percentage = count / len(self.processed_samples) * 100
            print(f"    {label:25s}: {count:6d} ({percentage:5.1f}%)")

    def _process_video_with_seed(self, match_id, parquet_file, events, seed, task_id):
        """
        wrapper to ensure deterministic seeding per task before processing.
        """
        task_seed = seed + task_id
        random.seed(task_seed)
        np.random.seed(task_seed)
        return self._process_video(match_id, parquet_file, events)
        
    def _process_video(self, match_id, parquet_file, events):
        """
        processes a single match parquet file and extracts event windows.
        aligns tracking frames with event timestamps using time tolerance matching.
        """
        df = pd.read_parquet(parquet_file).sort_values('videoTimeMs')
        samples = []
        matched = skipped = 0
        frames_with_annotations = set()

        for event in events:
            event_position_ms = event.get('position')
            time_diff = abs(df['videoTimeMs'] - event_position_ms)
            closest_idx = time_diff.idxmin()
            closest_row = df.loc[closest_idx]
            
            # skip events without tracking data within time tolerance
            # at 29.97 fps each frame is ~33ms apart, 10ms tolerance ensures accurate alignment
            if time_diff.loc[closest_idx] > self.time_tolerance_ms:
                skipped += 1
                continue

            matched += 1
            center_frame_num = int(closest_row['frameNum'])
            frames_with_annotations.add(center_frame_num)

            result = self._extract_window_features(df, center_frame_num)
            if result is None:
                matched -= 1
                skipped += 1
                continue
            
            features, positions = result

            # build edge indices for all frames during loading to avoid recomputation
            edge_indices = [
                self._build_edge_index_numpy(features[t], positions[t], edges=self.edge) 
                for t in range(self.window_size)
            ]

            samples.append({
                'window_id': f"{match_id}_{event_position_ms}_{event['label']}",
                'match_id': match_id,
                'label': event['label'],
                'game_time': event['gameTime'],
                'team': event.get('team'),
                'features': features,
                'edge_indices': edge_indices
            })

        return samples, matched, skipped, len(df), len(frames_with_annotations)

    def _extract_window_features(self, df, center_frame_num):
        """
        extracts a temporal window of features centered around the event frame.
        returns both node features and position labels for edge construction.
        """
        half_window = self.window_size // 2
        
        # apply random temporal jittering during training for data augmentation
        if self.random_positioning and self.split == 'train' and random.random() < self.random_positioning_prob:
            position_offset = random.randint(-self.max_position_shift, self.max_position_shift)
            half_window = half_window + position_offset
        
        frame_numbers = [
            center_frame_num + (i - half_window) * self.frame_interval 
            for i in range(self.window_size)
        ]
        
        # initialize feature array: (window_size, num_objects, feature_dim)
        features = np.zeros((self.window_size, self.num_objects, self.feature_dim), dtype=np.float32)
        
        # position labels for positional edge construction (GK, DEF, MID, FWD, BALL)
        positions = np.full((self.window_size, self.num_objects), '', dtype='U10')
        
        prev_positions_coord = {}

        for frame_idx, fnum in enumerate(frame_numbers):
            frame_data = df[df['frameNum'] == fnum]
            
            if frame_data.empty:
                # fill missing frames with sentinel values (-200.0)
                for obj_idx in range(self.num_objects):
                    features[frame_idx, obj_idx, :] = [-200.0, -200.0, 0, 0, 0, -200.0, -200.0, -200.0]
                    positions[frame_idx, obj_idx] = ''
                continue
            
            row = frame_data.iloc[0]
            obj_idx = 0

            # extract ball data (always first object at index 0)
            ball_str = row.get('balls', 'null')
            if pd.notna(ball_str) and ball_str not in ['null', '']: 
                ball_list = json.loads(ball_str) 
                if ball_list: 
                    ball = ball_list[0]
                    x, y, z = ball.get('x'), ball.get('y'), ball.get('z')
                    if x is not None and y is not None:
                        x, y, z = float(x), float(y), float(z)
                        dx = x - prev_positions_coord.get(obj_idx, (x, y))[0] if frame_idx > 0 else 0
                        dy = y - prev_positions_coord.get(obj_idx, (x, y))[1] if frame_idx > 0 else 0
                        features[frame_idx, obj_idx, :] = [x, y, 1, 0, 0, dx, dy, z]
                        prev_positions_coord[obj_idx] = (x, y)
                        positions[frame_idx, obj_idx] = 'BALL'
                    else:
                        features[frame_idx, obj_idx, :] = [-200.0, -200.0, 0, 0, 0, -200.0, -200.0, -200.0]
                        positions[frame_idx, obj_idx] = ''
                else:
                    features[frame_idx, obj_idx, :] = [-200.0, -200.0, 0, 0, 0, -200.0, -200.0, -200.0]
                    positions[frame_idx, obj_idx] = ''
            else:
                features[frame_idx, obj_idx, :] = [-200.0, -200.0, 0, 0, 0, -200.0, -200.0, -200.0]
                positions[frame_idx, obj_idx] = ''
            obj_idx += 1

            # extract player data for both teams
            for team_key, team_idx in [('homePlayers', 1), ('awayPlayers', 2)]:
                players_str = row[team_key]
                players = json.loads(players_str) if players_str and players_str != 'null' else []
                # sort by jersey number and take first 11 (handles rare >11 visible players)
                players = sorted(players, key=lambda p: int(p.get('jerseyNum', 0)))[:11]
                
                for player in players:
                    x, y = player.get('x'), player.get('y')
                    position_group = player.get('positionGroup', '')
                    
                    if x is not None and y is not None:
                        x, y = float(x), float(y)
                        team_one_hot = [0, 0, 0]
                        team_one_hot[team_idx] = 1
                        dx = x - prev_positions_coord.get(obj_idx, (x, y))[0] if frame_idx > 0 else 0
                        dy = y - prev_positions_coord.get(obj_idx, (x, y))[1] if frame_idx > 0 else 0
                        features[frame_idx, obj_idx, :] = [x, y] + team_one_hot + [dx, dy, -200.0]
                        positions[frame_idx, obj_idx] = position_group if position_group else ''
                        prev_positions_coord[obj_idx] = (x, y)
                    else:
                        features[frame_idx, obj_idx, :] = [-200.0, -200.0, 0, 0, 0, -200.0, -200.0, -200.0]
                        positions[frame_idx, obj_idx] = ''
                    obj_idx += 1
                
                # fill remaining slots if fewer than 11 players detected
                for _ in range(11 - len(players)):
                    features[frame_idx, obj_idx, :] = [-200.0, -200.0, 0, 0, 0, -200.0, -200.0, -200.0]
                    positions[frame_idx, obj_idx] = ''
                    obj_idx += 1

        return features, positions

    def _resplit_by_games(self, num_games_to_keep):
        """
        reduces dataset to only include samples from the first N matches.
        used for data scaling experiments.
        """
        if num_games_to_keep is None:
            return
        
        match_ids = sorted(set([s['match_id'] for s in self.processed_samples]))
        selected_matches = match_ids[:num_games_to_keep]
        self.processed_samples = [s for s in self.processed_samples if s['match_id'] in selected_matches]
        print(f"{self.split} resplit: {num_games_to_keep} games, {len(self.processed_samples)} samples")

    def _normalize_features(self, features):
        """
        normalizes features to [-1, 1] range, with -2.0 as sentinel for missing values.
        """
        features_norm = features.copy()
        
        valid_mask = features_norm[:, :, 0] != -200.0
        features_norm[valid_mask, 0] /= self.pitch_half_length
        features_norm[valid_mask, 1] /= self.pitch_half_width
        features_norm[valid_mask, 5] /= self.max_displacement  
        features_norm[valid_mask, 6] /= self.max_displacement
        features_norm[valid_mask, 7] /= self.max_ball_height
        
        # set missing values to -2.0 (outside normalized range) as sentinel
        features_norm[~valid_mask, 0] = -2.0
        features_norm[~valid_mask, 1] = -2.0
        features_norm[~valid_mask, 5] = -2.0
        features_norm[~valid_mask, 6] = -2.0
        features_norm[~valid_mask, 7] = -2.0

        return features_norm

    def _build_edge_index_numpy(self, node_features, node_positions, edges=None):
        """
        builds edge index for graph connectivity based on the specified scheme.
        
        supported edge types:
            none: no edges (independent nodes)
            full: fully connected graph
            ball_knn: k nearest players to ball, connected by team
            ball_distance: players within distance threshold of ball
            knn: k nearest neighbors for each node
            distance: nodes within distance threshold
            positional: tactical structure (GK->DEF->MID->FWD)
        """
        if edges == 'none':
            return np.zeros((2, 0), dtype=np.int64)
        
        num_nodes = node_features.shape[0]
        
        if edges == 'full': 
            edge_list = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
            return np.array(edge_list, dtype=np.int64).T if edge_list else np.zeros((2, 0), dtype=np.int64)
        
        if edges == 'ball_knn':
            ball_indices = np.where(node_features[:, 2] == 1.0)[0]
            if len(ball_indices) == 0:
                return np.zeros((2, 0), dtype=np.int64)
            
            ball_idx = ball_indices[0]
            ball_pos = node_features[ball_idx, :2]
            
            if ball_pos[0] == -200.0 or ball_pos[1] == -200.0:
                return np.zeros((2, 0), dtype=np.int64)
            
            # find k nearest players to ball
            distances = []
            for i in range(num_nodes):
                if i != ball_idx and (node_features[i, 3] == 1.0 or node_features[i, 4] == 1.0) and node_features[i, 0] != -200.0:
                    distances.append((i, np.linalg.norm(node_features[i, :2] - ball_pos)))
            
            distances.sort(key=lambda x: x[1])
            k_nearest = distances[:min(self.k, len(distances))] 
            
            edge_list = []
            for player_idx, _ in k_nearest:
                edge_list.extend([[ball_idx, player_idx], [player_idx, ball_idx]])
            
            # connect same-team players among k nearest
            k_nearest_indices = [idx for idx, _ in k_nearest]
            for i, idx_i in enumerate(k_nearest_indices):
                team_i = node_features[idx_i, 3:5]
                for j in range(i + 1, len(k_nearest_indices)):
                    idx_j = k_nearest_indices[j]
                    team_j = node_features[idx_j, 3:5]
                    if np.dot(team_i, team_j) > 0:
                        edge_list.extend([[idx_i, idx_j], [idx_j, idx_i]])
            
            return np.array(edge_list, dtype=np.int64).T if edge_list else np.zeros((2, 0), dtype=np.int64)

        if edges == 'ball_distance':
            # connect player within 20m of ball
            distance_threshold = 20.0
            ball_indices = np.where(node_features[:, 2] == 1.0)[0]
            if len(ball_indices) == 0:
                return np.zeros((2, 0), dtype=np.int64)
            
            ball_idx = ball_indices[0]
            ball_pos = node_features[ball_idx, :2]
            
            if ball_pos[0] == -200.0 or ball_pos[1] == -200.0:
                return np.zeros((2, 0), dtype=np.int64)
            
            edge_list = []
            nearby_players = []
            for i in range(num_nodes):
                if i != ball_idx and (node_features[i, 3] == 1.0 or node_features[i, 4] == 1.0) and node_features[i, 0] != -200.0:
                    dist = np.linalg.norm(node_features[i, :2] - ball_pos)
                    if dist <= distance_threshold:
                        edge_list.extend([[ball_idx, i], [i, ball_idx]])
                        nearby_players.append(i)
            
            # connect same team players
            for i, idx_i in enumerate(nearby_players):
                team_i = node_features[idx_i, 3:5]
                for j in range(i + 1, len(nearby_players)):
                    idx_j = nearby_players[j]
                    team_j = node_features[idx_j, 3:5]
                    if np.dot(team_i, team_j) > 0:
                        edge_list.extend([[idx_i, idx_j], [idx_j, idx_i]])
            
            return np.array(edge_list, dtype=np.int64).T if edge_list else np.zeros((2, 0), dtype=np.int64)
        
        if edges == 'knn':
            edge_list = []
            for i in range(num_nodes):
                if node_features[i, 0] == -200.0:
                    continue
                
                distances = []
                for j in range(num_nodes):
                    if i != j and node_features[j, 0] != -200.0:
                        dist = np.linalg.norm(node_features[i, :2] - node_features[j, :2])
                        distances.append((j, dist))
                
                distances.sort(key=lambda x: x[1])
                k_nearest = distances[:min(self.k, len(distances))]
                
                for neighbor_idx, _ in k_nearest:
                    edge_list.extend([[i, neighbor_idx], [neighbor_idx, i]])
            
            if not edge_list:
                return np.zeros((2, 0), dtype=np.int64)
            
            edge_array = np.array(edge_list, dtype=np.int64).T
            edge_array = np.unique(edge_array, axis=1)
            return edge_array
        
        if edges == 'distance':
            distance_threshold = 15.0
            edge_list = []
            for i in range(num_nodes):
                if node_features[i, 0] == -200.0:
                    continue
                for j in range(i + 1, num_nodes):
                    if node_features[j, 0] == -200.0:
                        continue
                    dist = np.linalg.norm(node_features[i, :2] - node_features[j, :2])
                    if dist <= distance_threshold:
                        edge_list.extend([[i, j], [j, i]])
            
            return np.array(edge_list, dtype=np.int64).T if edge_list else np.zeros((2, 0), dtype=np.int64)

        if edges == 'positional':
            # tactical structure edges connecting adjacent position groups
            edge_list = []
            
            home_players = {}
            away_players = {}
            
            for i in range(num_nodes):
                if node_features[i, 0] == -200.0 or not node_positions[i]:
                    continue
                
                pos = node_positions[i]
                if node_features[i, 3] == 1.0:
                    if pos not in home_players:
                        home_players[pos] = []
                    home_players[pos].append(i)
                elif node_features[i, 4] == 1.0:
                    if pos not in away_players:
                        away_players[pos] = []
                    away_players[pos].append(i)
            
            # build edges following tactical structure: GK <-> DEF <-> MID <-> FWD
            for team_players in [home_players, away_players]:
                gk = team_players.get('GK', [])
                defenders = team_players.get('DEF', [])
                midfielders = team_players.get('MID', [])
                forwards = team_players.get('FWD', [])
                
                for p1 in gk:
                    for p2 in defenders:
                        edge_list.extend([[p1, p2], [p2, p1]])
                
                for p1 in defenders:
                    for p2 in defenders:
                        if p1 != p2:
                            edge_list.extend([[p1, p2], [p2, p1]])
                    for p2 in midfielders:
                        edge_list.extend([[p1, p2], [p2, p1]])
                
                for p1 in midfielders:
                    for p2 in midfielders:
                        if p1 != p2:
                            edge_list.extend([[p1, p2], [p2, p1]])
                    for p2 in forwards:
                        edge_list.extend([[p1, p2], [p2, p1]])
                
                for p1 in forwards:
                    for p2 in forwards:
                        if p1 != p2:
                            edge_list.extend([[p1, p2], [p2, p1]])
            
            # ball connects to all players when present
            ball_idx = None
            for i in range(num_nodes):
                if node_features[i, 2] == 1.0 and node_features[i, 0] != -200.0:
                    ball_idx = i
                    break
            
            if ball_idx is not None:
                for i in range(num_nodes):
                    if i != ball_idx and node_features[i, 0] != -200.0:
                        edge_list.extend([[ball_idx, i], [i, ball_idx]])
            
            if not edge_list:
                return np.zeros((2, 0), dtype=np.int64)
            
            edge_array = np.array(edge_list, dtype=np.int64).T
            edge_array = np.unique(edge_array, axis=1)
            return edge_array
        
        # default: no edges
        return np.zeros((2, 0), dtype=np.int64)

    def __len__(self):
        return len(self.processed_samples)

    def num_classes(self):
        return len(self.labels)

    def get_classes(self):
        return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]

    def __getitem__(self, idx):
        sample = self.processed_samples[idx]
        features = sample['features'].copy()
        
        # apply spatial augmentations
        if self.transforms:
            for transform in self.transforms:
                features = transform(features)
        
        if self.normalize:
            features = self._normalize_features(features)
        
        # create graph data objects for each frame in the window
        graphs = []
        for t in range(features.shape[0]):
            data = Data(
                x=torch.tensor(features[t], dtype=torch.float),
                edge_index=torch.tensor(sample['edge_indices'][t], dtype=torch.long),
                time_idx=t
            )
            graphs.append(data)
        
        return {
            'graphs': graphs,
            'label': sample['label'],
            'seq_len': len(graphs),
            'window_id': sample['window_id'],
            'label_name': sample['label_name'],
            'game_time': sample['game_time'],
            'team': sample['team'],
            'match_id': sample['match_id']
        }


class HorizontalFlip:
    """
    randomly flips x coordinates horizontally (along pitch length).
    """
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, features):
        if random.random() < self.probability:
            features_flipped = features.copy()
            features_flipped[:, :, 0] = -features_flipped[:, :, 0]
            features_flipped[:, :, 5] = -features_flipped[:, :, 5]
            return features_flipped
        return features


class VerticalFlip:
    """
    randomly flips y coordinates vertically (along pitch width).
    """
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, features):
        if random.random() < self.probability:
            features_flipped = features.copy()
            features_flipped[:, :, 1] = -features_flipped[:, :, 1]
            features_flipped[:, :, 6] = -features_flipped[:, :, 6]
            return features_flipped
        
        return features


class TeamFlip:
    """
    randomly swaps home and away team labels.
    keeps all other features unchanged.
    """
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, features):
        if random.random() < self.probability:
            features_flipped = features.copy()
            home_team = features_flipped[:, :, 3].copy()
            features_flipped[:, :, 3] = features_flipped[:, :, 4]
            features_flipped[:, :, 4] = home_team
            return features_flipped
        
        return features