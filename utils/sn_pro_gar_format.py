import os
import json
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime
from collections import defaultdict

LABELS = [
    "PASS", "HEADER", "HIGH PASS", "OUT", "CROSS",
    "THROW IN", "SHOT", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"
]

# higher value = higher priority, kept when multiple labels share a timestamp
LABEL_PRIORITY = {
    "PASS": 0,
    "HIGH PASS": 1,
    "OUT": 2,
    "CROSS": 3,
    "SHOT": 4,
    "HEADER": 5,
    "PLAYER SUCCESSFUL TACKLE": 6,
    "THROW IN": 7,
    "FREE KICK": 8,
    "GOAL": 9,
}

TIME_TOLERANCE_MS = 10


def deduplicate_annotations(annotations):
    """
    safety net: if multiple annotations share the same position_ms,
    keep only the one with the highest priority label.
    """
    by_position = defaultdict(list)
    for ann in annotations:
        by_position[ann['position']].append(ann)

    deduped = []
    for position_ms in sorted(by_position.keys()):
        candidates = by_position[position_ms]
        if len(candidates) == 1:
            deduped.append(candidates[0])
        else:
            # pick the highest priority label
            best = max(candidates, key=lambda a: LABEL_PRIORITY.get(a['label'], -1))
            deduped.append(best)

    return deduped


def process_game(args):
    game_id, video_path, parquet_path, annotations, split, output_dir, window_size, frame_interval, modality = args

    cap = None
    fps = None
    if modality in ['frames', 'both']:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

    # always load tracking for alignment regardless of modality
    tracking_df = pd.read_parquet(parquet_path).sort_values(
        ['videoTimeMs', 'frameNum'], ascending=[True, True]
    ).reset_index(drop=True)

    results = []
    skipped = 0

    for ann in annotations:
        position_ms = ann['position']
        label = ann['label']

        if label not in LABELS:
            continue

        # find closest tracking frame by videoTimeMs
        time_diff = (tracking_df['videoTimeMs'] - position_ms).abs()
        closest_idx = time_diff.idxmin()

        # skip if no tracking data within tolerance
        if time_diff.loc[closest_idx] > TIME_TOLERANCE_MS:
            skipped += 1
            continue

        closest_row = tracking_df.loc[closest_idx]
        center_frame_num = int(closest_row['frameNum'])

        # build window centered on the event
        half_window = window_size // 2
        frame_numbers = [
            center_frame_num + (i - half_window) * frame_interval
            for i in range(window_size)
        ]

        # extract tracking clip rows by frameNum
        clip_rows = []
        for fnum in frame_numbers:
            frame_data = tracking_df[tracking_df['frameNum'] == fnum]
            if not frame_data.empty:
                clip_rows.append(frame_data.iloc[0])
            else:
                empty_row = pd.Series({
                    'videoTimeMs': np.nan,
                    'frameNum': fnum,
                    'period': -1,
                    'balls': '[]',
                    'homePlayers': '[]',
                    'awayPlayers': '[]',
                })
                clip_rows.append(empty_row)

        clip_df = pd.DataFrame(clip_rows).reset_index(drop=True)

        # video frame extraction
        frames = None
        if modality in ['frames', 'both']:
            center_frame = int((position_ms / 1000.0) * fps)
            start_frame = center_frame - (window_size // 2) * frame_interval

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

        results.append({
            'frames': frames,
            'tracking': clip_df,
            'label': label,
            'game_id': game_id,
            'game_time': ann['gameTime'],
            'position_ms': position_ms,
            'team': ann['team']
        })

    if cap is not None:
        cap.release()

    if skipped > 0:
        print(f"  {game_id}: skipped {skipped} events (time tolerance > {TIME_TOLERANCE_MS}ms)")

    return results


def print_split_stats(split, data_entries):
    """print per-class counts for a split."""
    counts = defaultdict(int)
    for entry in data_entries:
        counts[entry['labels']['action']['label']] += 1

    total = sum(counts.values())
    print(f"\n  {split} stats: {total} clips")
    for label in LABELS:
        c = counts[label]
        pct = (c / total * 100) if total > 0 else 0
        print(f"    {label}: {c} ({pct:.1f}%)")


def print_overall_stats(all_stats):
    """print combined stats across all splits."""
    print(f"\n{'='*50}")
    print("overall dataset stats")
    print('='*50)

    total_counts = defaultdict(int)
    grand_total = 0

    for split in ['train', 'valid', 'test']:
        if split in all_stats:
            for label, count in all_stats[split].items():
                total_counts[label] += count
                grand_total += count

    print(f"  total clips: {grand_total}")
    for label in LABELS:
        c = total_counts[label]
        pct = (c / grand_total * 100) if grand_total > 0 else 0
        print(f"    {label}: {c} ({pct:.1f}%)")


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

    all_stats = {}

    for split in ['train', 'valid', 'test']:
        print(f"\n{'='*50}")
        print(f"processing {split}")
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

            if not os.path.exists(parquet_path):
                print(f"skipping {game_id}: missing parquet (required for alignment)")
                continue

            if args.modality in ['frames', 'both'] and not os.path.exists(video_path):
                print(f"skipping {game_id}: missing video")
                continue

            # deduplicate annotations before passing to process_game
            raw_annotations = video_data.get('annotations', [])
            annotations = deduplicate_annotations(raw_annotations)

            if len(annotations) != len(raw_annotations):
                removed = len(raw_annotations) - len(annotations)
                print(f"  {game_id}: removed {removed} duplicate annotations via priority resolution")

            game_args.append((
                game_id, video_path, parquet_path, annotations,
                split, args.output_dir, args.window_size, args.frame_interval,
                args.modality
            ))

        # process games
        all_results = []
        with Pool(args.num_workers) as pool:
            for results in tqdm(pool.imap(process_game, game_args), total=len(game_args), desc=f"{split}"):
                all_results.extend(results)

        # save clips and build annotation entries
        data_entries = []
        for idx, result in enumerate(tqdm(all_results, desc="saving")):
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

        # collect and print per-split stats
        split_counts = defaultdict(int)
        for entry in data_entries:
            split_counts[entry['labels']['action']['label']] += 1
        all_stats[split] = dict(split_counts)

        print_split_stats(split, data_entries)

    # print combined stats
    print_overall_stats(all_stats)


if __name__ == '__main__':
    main()
    