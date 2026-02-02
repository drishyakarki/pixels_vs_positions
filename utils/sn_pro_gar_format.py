import os
import json
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime

LABELS = [
    "PASS", "HEADER", "HIGH PASS", "OUT", "CROSS",
    "THROW IN", "SHOT", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"
]


def process_game(args):
    game_id, video_path, parquet_path, annotations, split, output_dir, window_size, frame_interval = args
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    tracking_df = pd.read_parquet(parquet_path)
    
    results = []
    
    for ann in annotations:
        position_ms = ann['position']
        label = ann['label']
        
        if label not in LABELS:
            continue
        
        center_frame = int((position_ms / 1000.0) * fps)
        start_frame = center_frame - (window_size // 2) * frame_interval
        
        # extract video frames
        frames = []
        for i in range(window_size):
            frame_idx = start_frame + i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        frames = np.array(frames)
        
        # extract tracking clip
        frame_indices = [start_frame + i * frame_interval for i in range(window_size)]
        frame_times_ms = [(f / fps) * 1000 for f in frame_indices]
        
        clip_rows = []
        for target_ms in frame_times_ms:
            idx = (tracking_df['videoTimeMs'] - target_ms).abs().idxmin()
            clip_rows.append(tracking_df.loc[idx])
        
        clip_df = pd.DataFrame(clip_rows).reset_index(drop=True)
        
        results.append({
            'frames': frames,
            'tracking': clip_df,
            'label': label,
            'game_id': game_id,
            'game_time': ann['gameTime'],
            'position_ms': position_ms,
            'team': ann['team']
        })
    
    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', default='soccernetpro-classification-GAR/video_dataset')
    parser.add_argument('--tracking-dir', default='soccernetpro-classification-GAR/tracking_dataset')
    parser.add_argument('--output-dir', default='soccernetpro-classification-GAR/soccernet_gar')
    parser.add_argument('--modality', choices=['frames', 'tracking', 'both'], default='both')
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--frame-interval', type=int, default=9)
    parser.add_argument('--num-workers', type=int, default=30)
    args = parser.parse_args()
    
    for split in ['train', 'valid', 'test']:
        print(f"\n{'='*50}")
        print(f"Processing {split}")
        print('='*50)
        
        if args.modality == 'frames':
            frames_out = os.path.join(args.output_dir, split)
            os.makedirs(frames_out, exist_ok=True)
        elif args.modality == 'tracking':
            tracking_out = os.path.join(args.output_dir, split)
            os.makedirs(tracking_out, exist_ok=True)
        else:
            frames_out = os.path.join(args.output_dir, 'frames_npy', split)
            tracking_out = os.path.join(args.output_dir, 'tracking_parquet', split)
            os.makedirs(frames_out, exist_ok=True)
            os.makedirs(tracking_out, exist_ok=True)
        
        manifest_path = os.path.join(args.video_dir, split, f'{split}.json')
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # prepare args for each game
        game_args = []
        for video_data in manifest['videos']:
            game_id = video_data['gameId']
            video_path = os.path.join(args.video_dir, split, 'videos', f'{game_id}.mp4')
            parquet_path = os.path.join(args.tracking_dir, split, 'videos', f'{game_id}.parquet')
            
            if not os.path.exists(video_path) or not os.path.exists(parquet_path):
                print(f"Skipping {game_id}: missing files")
                continue
            
            annotations = video_data.get('annotations', [])
            game_args.append((
                game_id, video_path, parquet_path, annotations,
                split, args.output_dir, args.window_size, args.frame_interval
            ))
        
        # process games in parallel
        all_results = []
        with Pool(args.num_workers) as pool:
            for results in tqdm(pool.imap(process_game, game_args), total=len(game_args), desc=f"{split}"):
                all_results.extend(results)
        
        # save clips and build annotation
        data_entries = []
        for idx, result in enumerate(tqdm(all_results, desc="Saving")):
            frames_filename = f'clip_{idx:06d}.npy'
            tracking_filename = f'clip_{idx:06d}.parquet'
            
            if args.modality == 'frames':
                np.save(os.path.join(frames_out, frames_filename), result['frames'])
            elif args.modality == 'tracking':
                result['tracking'].to_parquet(os.path.join(tracking_out, tracking_filename), index=False)
            else:
                np.save(os.path.join(frames_out, frames_filename), result['frames'])
                result['tracking'].to_parquet(os.path.join(tracking_out, tracking_filename), index=False)
            
            inputs = []
            if args.modality == 'frames':
                inputs.append({'type': 'frames_npy', 'path': f'{split}/{frames_filename}'})
            elif args.modality == 'tracking':
                inputs.append({'type': 'tracking_parquet', 'path': f'{split}/{tracking_filename}'})
            else:
                inputs.append({'type': 'frames_npy', 'path': f'frames_npy/{split}/{frames_filename}'})
                inputs.append({'type': 'tracking_parquet', 'path': f'tracking_parquet/{split}/{tracking_filename}'})
            
            data_entries.append({
                'id': f'{split}_{idx:06d}',
                'inputs': inputs,
                'labels': {
                    'action': {'label': result['label']}
                },
                'metadata': {
                    'game_id': result['game_id'],
                    'game_time': result['game_time'],
                    'position_ms': result['position_ms'],
                    'team': result['team']
                }
            })
        
        # save annotation file
        annotation = {
            "version": "2.0",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "task": "action_classification",
            "modalities": (
                ["frames_npy"] if args.modality == 'frames' else
                ["tracking_parquet"] if args.modality == 'tracking' else
                ["frames_npy", "tracking_parquet"]
            ),
            "dataset_name": f"soccernet_gar_{'multimodal' if args.modality == 'both' else args.modality}_{split}",
            "labels": {
                "action": {
                    "type": "single_label",
                    "labels": LABELS
                }
            },
            "data": data_entries
        }
        
        ann_path = os.path.join(args.output_dir, f"annotations_{split}.json")
        with open(ann_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"{split}: {len(data_entries)} clips")


if __name__ == '__main__':
    main()