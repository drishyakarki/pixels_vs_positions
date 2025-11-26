"""
converts PFF FC dataset to SoccerNet-GAR format.

usage:
    # for tracking data (jsonl.bz2 -> parquet)
    python utils/convert_pff_to_soccernet.py --modality tracking \
        --events-dir data/events \
        --tracking-dir data/tracking \
        --output-dir data/tracking_dataset

    # for video data (copies mp4 files)
    python utils/convert_pff_to_soccernet.py --modality video \
        --events-dir data/events \
        --video-dir data/224p \
        --output-dir data/video_dataset
"""

import os
import json
import argparse
import bz2
import shutil
from collections import defaultdict
import concurrent.futures
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm


# dataset splits: (start_idx, end_idx) for sorting games alphabetically
SPLITS = {
    "train": (0, 45),
    "valid": (45, 54),
    "test": (54, 64),
}

# activity classes for group activity recognition
LABELS = [
    "PASS", "HEADER", "HIGH PASS", "OUT", "CROSS",
    "THROW IN", "SHOT", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"
]

# mapping from detailed positions to position groups
POSITION_GROUPS = {
    'GK': 'GK',
    'LCB': 'DEF', 'RCB': 'DEF', 'MCB': 'DEF', 'LB': 'DEF', 'RB': 'DEF', 'LWB': 'DEF', 'RWB': 'DEF',
    'CM': 'MID', 'AM': 'MID', 'DM': 'MID', 'LM': 'MID', 'RM': 'MID',
    'CF': 'FWD', 'LW': 'FWD', 'RW': 'FWD'
}


def build_position_mapping(jsonl_path):
    """
    extracts player position mappings from game events in the tracking file.
    returns dict mapping team_id -> jersey_num -> position.
    """
    position_map = defaultdict(lambda: defaultdict(lambda: None))

    with bz2.open(jsonl_path, 'rt') as f:
        for line in f:
            try:
                frame = json.loads(line)
                game_event = frame.get('game_event')

                if game_event and isinstance(game_event, dict):
                    team_id = game_event.get('team_id')
                    shirt_num = game_event.get('shirt_number')
                    position = game_event.get('position_group_type')

                    if team_id and shirt_num and position:
                        position_map[str(team_id)][str(shirt_num)] = position
            except:
                continue

    return dict(position_map)


def add_positions_to_players(players_list, team_id, position_map):
    """
    adds position and positionGroup fields to each player dict.
    """
    if not players_list or not isinstance(players_list, list):
        return players_list

    team_positions = position_map.get(str(team_id), {})

    for player in players_list:
        if isinstance(player, dict):
            jersey = str(player.get('jerseyNum', ''))
            position = team_positions.get(jersey)
            player['position'] = position if position else None
            player['positionGroup'] = POSITION_GROUPS.get(position) if position else None

    return players_list


def flatten_frame(frame, position_map):
    """
    flattens a single frame from nested json to flat dict for parquet storage.
    """
    flat = {
        'videoTimeMs': frame.get('videoTimeMs'),
        'frameNum': frame.get('frameNum'),
        'period': frame.get('period'),
        'game_event_id': frame.get('game_event_id'),
        'possession_event_id': frame.get('possession_event_id')
    }

    game_event = frame.get('game_event', {})
    home_team_id = None
    away_team_id = None

    if isinstance(game_event, dict):
        for key in ['game_event_type', 'player_name', 'player_id', 'team_id', 'home_team', 'video_url']:
            flat[key] = game_event.get(key, "")

        if game_event.get('home_team'):
            home_team_id = game_event.get('team_id')
        else:
            away_team_id = game_event.get('team_id')
    else:
        for key in ['game_event_type', 'player_name', 'player_id', 'team_id', 'home_team', 'video_url']:
            flat[key] = ""

    possession_event = frame.get('possession_event', {})
    if isinstance(possession_event, dict):
        flat['possession_event_type'] = possession_event.get('possession_event_type', "")
    else:
        flat['possession_event_type'] = ""

    # add player positions
    home_players = frame.get('homePlayers', [])
    away_players = frame.get('awayPlayers', [])

    if home_team_id:
        home_players = add_positions_to_players(home_players, home_team_id, position_map)
    if away_team_id:
        away_players = add_positions_to_players(away_players, away_team_id, position_map)

    # store player data as json strings
    flat['homePlayers'] = json.dumps(home_players if home_players else [])
    flat['homePlayersSmoothed'] = json.dumps(frame.get('homePlayersSmoothed', []))
    flat['awayPlayers'] = json.dumps(away_players if away_players else [])
    flat['awayPlayersSmoothed'] = json.dumps(frame.get('awayPlayersSmoothed', []))
    flat['balls'] = json.dumps(frame.get('balls', []))
    flat['ballsSmoothed'] = json.dumps(frame.get('ballsSmoothed', []))

    return flat


def convert_jsonl_to_parquet(jsonl_path, parquet_path):
    """
    converts a compressed jsonl tracking file to parquet format.
    """
    position_map = build_position_mapping(jsonl_path)
    frames = []

    with bz2.open(jsonl_path, 'rt') as f:
        for line in f:
            try:
                frame = json.loads(line)
                frames.append(flatten_frame(frame, position_map))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(frames)

    # convert column types
    int_columns = ['frameNum', 'period', 'game_event_id', 'possession_event_id']
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype('int32')

    float_columns = ['videoTimeMs']
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

    string_columns = [col for col in df.columns if col not in int_columns + float_columns]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    df.to_parquet(parquet_path, engine='pyarrow', compression=None, index=False)
    return len(frames)


def process_tracking_file(args):
    """
    worker function for parallel tracking file conversion.
    """
    idx, tracking_file, tracking_dir, output_dir = args
    game_id = tracking_file.replace('.jsonl.bz2', '')

    split_name = None
    for split, (start, end) in SPLITS.items():
        if start <= idx < end:
            split_name = split
            break

    if split_name:
        src = os.path.join(tracking_dir, tracking_file)
        dst = os.path.join(output_dir, split_name, "videos", f"{game_id}.parquet")

        if not os.path.exists(dst):
            convert_jsonl_to_parquet(src, dst)

        return (game_id, split_name)

    return None


def extract_annotations(events_path):
    """
    extracts activity annotations from pff event json file.
    maps pff event types to soccernet-gar labels.
    """
    with open(events_path, 'r') as f:
        data = json.load(f)

    annotations = []

    for event in data:
        labels = []

        possession = event.get('possessionEvents', {})
        event_type = possession.get('possessionEventType', '')
        body_type = possession.get('bodyType', '')

        if event_type == 'PA':
            if body_type == 'HE':
                labels.append('HEADER')
            elif possession.get('ballHeightType') == 'A':
                labels.append('HIGH PASS')
            elif possession.get('passType') == "H":
                labels.append("THROW IN")
            else:
                labels.append('PASS')

        elif event_type == 'CR':
            labels.append('CROSS')

        elif event_type == 'SH':
            if body_type == 'HE':
                labels.append('HEADER')
            labels.append('SHOT')
            if possession.get('shotOutcomeType') == 'G':
                labels.append('GOAL')

        elif event_type == 'CH':
            if possession.get('challengeWinnerPlayerId'):
                labels.append('PLAYER SUCCESSFUL TACKLE')

        elif event_type == 'CL' and body_type == 'HE':
            labels.append('HEADER')

        game_events = event.get('gameEvents', {})
        if game_events.get('gameEventType') == 'OUT':
            labels.append('OUT')

        setpiece = game_events.get('setpieceType', '')
        if setpiece == 'T':
            labels.append('THROW IN')
        elif setpiece == 'F':
            labels.append('FREE KICK')

        # create annotation entry for each valid label
        for label in labels:
            if label in LABELS:
                game_events = event.get('gameEvents', {})
                annotation = {
                    "gameTime": f"{game_events.get('period', 1)} - {game_events.get('startFormattedGameClock', '00:00')}",
                    "label": label,
                    "position": int(event.get('eventTime', 0) * 1000),
                    "team": "home" if game_events.get('homeTeam', False) else "away",
                    "visibility": "visible",
                }
                annotations.append(annotation)

    return sorted(annotations, key=lambda x: x["position"])


def create_split_manifests(events_dir, file_mapping, output_dir, file_format):
    """
    creates train.json, valid.json, test.json manifest files for each split.
    """
    split_data = {split: [] for split in SPLITS}

    print("Creating split manifests...")
    for game_id, split in tqdm(file_mapping.items()):
        file_ext = ".parquet" if file_format == "parquet" else ".mp4"

        video_entry = {
            "path": f"videos/{game_id}{file_ext}",
            "input_type": file_format,
            "gameId": game_id,
        }

        events_path = os.path.join(events_dir, f"{game_id}.json")
        if os.path.exists(events_path):
            video_entry["annotations"] = extract_annotations(events_path)

        split_data[split].append(video_entry)

    # write manifest files
    for split, videos in split_data.items():
        if videos:
            output_file = os.path.join(output_dir, split, f"{split}.json")
            split_json = {
                "version": 1,
                "format": file_format,
                "fps": 30,
                "videos": videos,
                "labels": LABELS,
            }
            with open(output_file, 'w') as f:
                json.dump(split_json, f, indent=2)

    return split_data


def process_tracking_modality(events_dir, tracking_dir, output_dir):
    """
    converts tracking data from jsonl.bz2 to parquet format and creates manifests.
    """
    if os.path.exists(os.path.join(output_dir, "train", "train.json")):
        print("Tracking dataset already exists. Skipping conversion.")
        return

    print("Converting tracking data to parquet format...")

    for split in SPLITS:
        os.makedirs(os.path.join(output_dir, split, "videos"), exist_ok=True)

    # convert tracking files in parallel
    tracking_files = sorted([f for f in os.listdir(tracking_dir) if f.endswith('.jsonl.bz2')])
    args_list = [(idx, f, tracking_dir, output_dir) for idx, f in enumerate(tracking_files)]

    file_mapping = {}
    max_workers = max(1, mp.cpu_count() - 2)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_tracking_file, args) for args in args_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Converting"):
            result = future.result()
            if result:
                game_id, split_name = result
                file_mapping[game_id] = split_name

    split_data = create_split_manifests(events_dir, file_mapping, output_dir, "parquet")

    print(f"\nTracking dataset created at: {output_dir}")
    for split in SPLITS:
        print(f"  {split}: {len(split_data[split])} videos")


def process_video_modality(events_dir, video_dir, output_dir):
    """
    copies video files to output directory and creates manifests.
    """
    if os.path.exists(os.path.join(output_dir, "train", "train.json")):
        print("Video dataset already exists. Skipping copy.")
        return

    print("Copying video files...")

    for split in SPLITS:
        os.makedirs(os.path.join(output_dir, split, "videos"), exist_ok=True)

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    file_mapping = {}

    for idx, video_file in enumerate(tqdm(video_files, desc="Copying")):
        game_id = video_file.replace('.mp4', '')

        split_name = None
        for split, (start, end) in SPLITS.items():
            if start <= idx < end:
                split_name = split
                break

        if split_name:
            src = os.path.join(video_dir, video_file)
            dst = os.path.join(output_dir, split_name, "videos", video_file)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)

            file_mapping[game_id] = split_name

    split_data = create_split_manifests(events_dir, file_mapping, output_dir, "mp4")

    print(f"\nVideo dataset created at: {output_dir}")
    for split in SPLITS:
        print(f"  {split}: {len(split_data[split])} videos")


def main():
    parser = argparse.ArgumentParser(description="Convert PFF FC dataset to SoccerNet-GAR format")
    parser.add_argument("--modality", choices=["tracking", "video"], required=True)
    parser.add_argument("--events-dir", default="data/events")
    parser.add_argument("--tracking-dir", default="data/tracking")
    parser.add_argument("--video-dir", default="data/224p")
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    if args.modality == "tracking":
        output_dir = args.output_dir if args.output_dir else "data/tracking_dataset"
        process_tracking_modality(args.events_dir, args.tracking_dir, output_dir)

    elif args.modality == "video":
        output_dir = args.output_dir if args.output_dir else "data/video_dataset"
        process_video_modality(args.events_dir, args.video_dir, output_dir)


if __name__ == "__main__":
    main()