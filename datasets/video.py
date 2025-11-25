import json
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time


def preprocess_split(data_dir, split, output_dir, window_size=16, frame_interval=9):
    """
    extracts clips from videos and saves them in .npy format for efficient loading during training.
    each split (train/valid/test) is preprocessed separately.
    
    the clips are centered around annotated events, with a temporal window determined by
    window_size and frame_interval parameters.
    """
    start_time = time.time()
    ann_path = os.path.join(data_dir, split, f'{split}.json')
    video_dir = os.path.join(data_dir, split, 'videos')
    
    with open(ann_path, 'r') as f:
        data = json.load(f)
    
    labels = data['labels']
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    clips = []
    
    for video_data in data['videos']:
        video_path = os.path.join(video_dir, os.path.basename(video_data['path']))
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        for ann in video_data.get('annotations', []):
            # we get the time in milliseconds and convert it to frame index.
            # the center frame is where the event occurs, and we build a window around it.
            position_ms = ann['position']
            label = ann['label']
            
            center_frame = int((position_ms / 1000.0) * fps)
            start_frame = center_frame - (window_size // 2) * frame_interval
            
            clips.append({
                'video_path': video_path,
                'start_frame': start_frame,
                'label': label,
                'label_idx': label_to_idx[label],
                'position_ms': position_ms
            })
    
    print(f"\n{split}: {len(clips)} clips")
    
    os.makedirs(output_dir, exist_ok=True)

    # use multiprocessing to extract clips in parallel for faster preprocessing
    num_workers = min(20, cpu_count())
    args_list = [(i, clips[i], window_size, frame_interval, output_dir) for i in range(len(clips))]
    
    with Pool(num_workers) as pool:
        clip_list = list(tqdm(
            pool.imap(_extract_single_clip, args_list, chunksize=10),
            total=len(clips),
            desc=f"Processing {split}"
        ))
    
    # save clip metadata to json for later loading by the dataset class
    with open(os.path.join(output_dir, 'clips.json'), 'w') as f:
        json.dump({
            'clips': clip_list,
            'labels': labels,
            'split': split
        }, f)
    
    elapsed = time.time() - start_time
    print(f"Completed {split}: {len(clips)} clips in {elapsed/60:.1f} minutes")


def _extract_single_clip(args):
    """
    extracts a single clip from a video file and saves it as a numpy array.
    this function is called in parallel by preprocess_split().
    """
    idx, clip_info, window_size, frame_interval, output_dir = args
    
    cap = cv2.VideoCapture(clip_info['video_path'])
    frames = []
    
    for i in range(window_size):
        frame_idx = clip_info['start_frame'] + i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        else:
            # if frame is not available, add a black frame as placeholder
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    cap.release()
    
    frames = np.array(frames)
    clip_filename = f"clip_{idx:06d}.npy"
    np.save(os.path.join(output_dir, clip_filename), frames)
    
    return {
        'filename': clip_filename,
        'label': clip_info['label'],
        'label_idx': clip_info['label_idx']
    }


class VideoDataset(Dataset):
    """
    dataset for loading preprocessed video clips for group activity recognition.
    
    clips are loaded on-demand from disk to avoid memory issues with large datasets.
    supports data augmentation with horizontal flipping and color jittering.
    """
    
    def __init__(self, preprocessed_dir, split='train', augment=False):
        self.split = split
        self.preprocessed_dir = preprocessed_dir
        
        # augmentation is only applied during training
        self.augment = augment if split == 'train' else False
        if self.augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        else:
            self.color_jitter = None
        
        clips_path = os.path.join(preprocessed_dir, split, 'clips.json')
        with open(clips_path, 'r') as f:
            data = json.load(f)
        
        self.all_labels = data['labels']
        all_clips = data['clips']
        
        self.labels = self.all_labels
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.clips = all_clips
        
        # remap label indices to match current label_to_idx after potential filtering
        for clip in self.clips:
            clip['label_idx'] = self.label_to_idx[clip['label']]

        print(f"{split}: {len(self.clips)} clips")
    
    def _resplit_by_games(self, num_games_to_keep, data_dir):
        """
        reduces the dataset to only include clips from the first N games.
        used for data scaling experiments to measure performance vs training data size.
        """
        if num_games_to_keep is None:
            return
        
        ann_path = os.path.join(data_dir, self.split, f'{self.split}.json')
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        # count clips per video to determine how many to keep
        clips_per_video = []
        for video_data in data['videos']:
            count = len(video_data.get('annotations', []))
            clips_per_video.append(count)
        
        total_clips_to_keep = sum(clips_per_video[:num_games_to_keep])
        self.clips = self.clips[:total_clips_to_keep]
        
        print(f"{self.split} resplit: {num_games_to_keep} games, {len(self.clips)} clips")

    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip = self.clips[idx]
        
        # load clip on-demand from disk to avoid memory issues
        clip_path = os.path.join(self.preprocessed_dir, self.split, clip['filename'])
        frames = np.load(clip_path).astype(np.float32) / 255.0
        
        # apply augmentations before normalization
        if self.augment:
            # horizontal flip with 50% probability
            if np.random.random() > 0.5:
                frames = frames[:, :, ::-1, :].copy()  
            
            # color jitter with 50% probability for robustness to lighting/color variations
            if np.random.random() > 0.5:
                frames_list = []
                for t in range(frames.shape[0]):
                    frame_t = torch.from_numpy(frames[t])
                    frame_t = frame_t.permute(2, 0, 1)
                    frame_t = self.color_jitter(frame_t)
                    frame_t = frame_t.permute(1, 2, 0)
                    frames_list.append(frame_t.numpy())
                frames = np.stack(frames_list)
        
        # normalize using ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames = (frames - mean[None, None, None, :]) / std[None, None, None, :]
        
        return frames, clip['label_idx']
    
    def num_classes(self):
        return len(self.labels)
    
    def get_classes(self):
        return self.labels
        