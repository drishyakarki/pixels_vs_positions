import os
import io
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from PIL import Image as PILImage, ImageDraw, ImageFont

LABELS = [
    "PASS", "HEADER", "HIGH PASS", "OUT", "CROSS",
    "THROW IN", "SHOT", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"
]

# pitch colors
LIGHT_GREEN = "#6da942"
DARK_GREEN = "#507d2a"
HOME_COLOR = "#CC0000"
AWAY_COLOR = "#0066CC"
BALL_COLOR = "purple"


# ---------------------------------------------------------------------------
# pitch drawing
# ---------------------------------------------------------------------------

def draw_pitch(ax, length=105.0, width=68.0):
    """draw a soccer pitch on the given axes."""
    padding = 2.0

    # background
    bg = Rectangle((-padding, -padding), length + 2 * padding, width + 2 * padding,
                   color=LIGHT_GREEN, zorder=0)
    ax.add_patch(bg)

    # stripes
    num_stripes = 20
    sw = length / num_stripes
    for i in range(num_stripes):
        color = LIGHT_GREEN if i % 2 == 0 else DARK_GREEN
        stripe = Rectangle((i * sw, 0), sw, width, color=color, zorder=0)
        ax.add_patch(stripe)

    # grass texture
    np.random.seed(42)
    noise = gaussian_filter(np.random.rand(200, 200), sigma=0.5)
    ax.imshow(noise, extent=(0, length, 0, width), alpha=0.03, zorder=1, cmap='gray')

    lc = 'white'
    lw = 2

    # boundary
    ax.plot([0, 0, length, length, 0], [0, width, width, 0, 0], color=lc, linewidth=lw)

    # halfway line
    ax.plot([length / 2, length / 2], [0, width], color=lc, linewidth=lw)

    # center circle and spot
    ax.add_patch(Circle((length / 2, width / 2), 9.15, color=lc, fill=False, linewidth=lw))
    ax.plot(length / 2, width / 2, 'o', color=lc, markersize=5)

    # penalty areas
    for x_start in [0, length - 16.5]:
        ax.add_patch(Rectangle((x_start, width / 2 - 20.15), 16.5, 40.3,
                               edgecolor=lc, fill=False, linewidth=lw))

    # goal areas
    for x_start in [0, length - 5.5]:
        ax.add_patch(Rectangle((x_start, width / 2 - 8.5), 5.5, 17,
                               edgecolor=lc, fill=False, linewidth=lw))

    # penalty arcs
    ax.add_patch(Arc((11, width / 2), 18.3, 18.3, theta1=308, theta2=52, color=lc, linewidth=lw))
    ax.add_patch(Arc((length - 11, width / 2), 18.3, 18.3, theta1=127, theta2=233, color=lc, linewidth=lw))

    # penalty spots
    ax.plot(11, width / 2, 'o', color=lc, markersize=5)
    ax.plot(length - 11, width / 2, 'o', color=lc, markersize=5)

    # goalposts
    ax.plot([0, 0], [width / 2 - 3.66, width / 2 + 3.66], color=lc, linewidth=lw + 2)
    ax.plot([length, length], [width / 2 - 3.66, width / 2 + 3.66], color=lc, linewidth=lw + 2)

    # corner arcs
    for x, y, t1, t2 in [(0, 0, 0, 90), (0, width, 270, 360),
                          (length, 0, 90, 180), (length, width, 180, 270)]:
        ax.add_patch(Arc((x, y), 1.8, 1.8, theta1=t1, theta2=t2, color=lc, linewidth=lw))

    ax.set_xlim(-padding, length + padding)
    ax.set_ylim(-padding, width + padding)
    ax.set_aspect('equal')
    ax.axis('off')


# ---------------------------------------------------------------------------
# tracking frame rendering
# ---------------------------------------------------------------------------

def parse_players(json_str):
    """parse player list from json string, return list of dicts with x, y, jerseyNum."""
    try:
        players = json.loads(json_str) if isinstance(json_str, str) else json_str
        if not isinstance(players, list):
            return []
        return [p for p in players if isinstance(p, dict) and p.get('x') is not None and p.get('y') is not None]
    except:
        return []


def parse_ball(json_str):
    """parse ball from json string, return dict with x, y or None."""
    try:
        balls = json.loads(json_str) if isinstance(json_str, str) else json_str
        if isinstance(balls, dict) and balls.get('x') is not None:
            return balls
        if isinstance(balls, list):
            for b in balls:
                if isinstance(b, dict) and b.get('x') is not None:
                    return b
        return None
    except:
        return None


def draw_tracking_frame(ax, row):
    """draw a single tracking frame on the pitch axes."""
    draw_pitch(ax)

    # parse data from the parquet row
    home_players = parse_players(row.get('homePlayers', '[]'))
    away_players = parse_players(row.get('awayPlayers', '[]'))
    ball = parse_ball(row.get('balls', '[]'))

    # draw home players
    for p in home_players:
        px = p['x'] + 52.5
        py = p['y'] + 34.0
        ax.add_patch(Circle((px, py), 1.0, facecolor=HOME_COLOR, edgecolor='black', linewidth=0.5, zorder=4))
        jersey = p.get('jerseyNum', '')
        if jersey:
            ax.text(px, py, str(int(jersey)), color='white', ha='center', va='center',
                    fontsize=7, fontweight='bold', zorder=5)

    # draw away players
    for p in away_players:
        px = p['x'] + 52.5
        py = p['y'] + 34.0
        ax.add_patch(Circle((px, py), 1.0, facecolor=AWAY_COLOR, edgecolor='black', linewidth=0.5, zorder=4))
        jersey = p.get('jerseyNum', '')
        if jersey:
            ax.text(px, py, str(int(jersey)), color='white', ha='center', va='center',
                    fontsize=7, fontweight='bold', zorder=5)

    # draw ball
    if ball:
        bx = ball['x'] + 52.5
        by = ball['y'] + 34.0
        ax.add_patch(Circle((bx, by), 0.6, facecolor=BALL_COLOR, edgecolor='black', linewidth=0.5, zorder=6))


# ---------------------------------------------------------------------------
# main logic
# ---------------------------------------------------------------------------

def find_clip(annotations, label, index):
    """find a clip by label and occurrence index."""
    count = 0
    for entry in annotations['data']:
        if entry['labels']['action']['label'] == label:
            if count == index:
                return entry
            count += 1
    return None


def load_tracking_clip(tracking_dir, split, clip_path):
    """load a tracking parquet clip."""
    # clip_path is like "train/clip_000042.parquet" or "tracking_parquet/train/clip_000042.parquet"
    # but the actual file lives at tracking_dir/split/clip_XXXXXX.parquet
    filename = os.path.basename(clip_path)
    full_path = os.path.join(tracking_dir, split, filename)
    if not os.path.exists(full_path):
        # try the path as-is relative to tracking_dir
        full_path = os.path.join(tracking_dir, clip_path)
    return pd.read_parquet(full_path)


def load_video_clip(video_dir, split, clip_path):
    """load a video npy clip."""
    filename = os.path.basename(clip_path)
    full_path = os.path.join(video_dir, split, filename)
    if not os.path.exists(full_path):
        full_path = os.path.join(video_dir, clip_path)
    return np.load(full_path)


def fig_to_pil(fig):
    """render a matplotlib figure to a PIL image using savefig to memory buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02)
    buf.seek(0)
    img = PILImage.open(buf).convert('RGB')
    plt.close(fig)
    return img


def render_tracking_to_pil(row, target_width=600):
    """render a single tracking frame to a PIL image."""
    # taller figure for bigger tracking cells
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=120)
    draw_tracking_frame(ax, row)
    img = fig_to_pil(fig)
    # resize to target width, keep aspect
    w, h = img.size
    target_height = int(target_width * h / w)
    return img.resize((target_width, target_height), PILImage.LANCZOS)


def save_side_by_side_frames(tracking_df, video_frames, label, game_time, output_path,
                              frame_indices=None):
    """save selected frames by rendering each cell to PIL image and stitching."""
    n_frames = len(tracking_df)

    if frame_indices is None:
        frame_indices = np.linspace(0, n_frames - 1, 4, dtype=int)

    n = len(frame_indices)
    cell_w = 500

    # render all tracking frames
    t_imgs = []
    for fi in frame_indices:
        img = render_tracking_to_pil(tracking_df.iloc[fi], target_width=cell_w)
        t_imgs.append(img)

    # render all video frames (resize to same width, shorter height)
    v_imgs = []
    for fi in frame_indices:
        img = PILImage.fromarray(video_frames[fi])
        # make video cells shorter: 70% of the natural square height
        target_h = int(cell_w * 0.56)
        img = img.resize((cell_w, target_h), PILImage.LANCZOS)
        v_imgs.append(img)

    t_cell_h = t_imgs[0].size[1]
    v_cell_h = v_imgs[0].size[1]

    gap = 3
    label_w = 90
    title_h = 50

    total_w = label_w + n * cell_w + (n - 1) * gap
    total_h = title_h + v_cell_h + gap + t_cell_h

    # build canvas
    canvas = PILImage.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
        font_frame = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font_title = ImageFont.load_default()
        font_label = font_title
        font_frame = font_title

    # title
    title_text = f"{label} at {game_time}"
    bbox = draw.textbbox((0, 0), title_text, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 20), title_text, fill=(0, 0, 0), font=font_title)

    # paste video row (top)
    y_t = title_h
    for col, img in enumerate(v_imgs):
        x = label_w + col * (cell_w + gap)
        canvas.paste(img, (x, y_t))

    # paste tracking row (bottom)
    y_v = title_h + v_cell_h + gap
    for col, img in enumerate(t_imgs):
        x = label_w + col * (cell_w + gap)
        canvas.paste(img, (x, y_v))

    # row labels (rotated text)
    for text, y_center in [("Video", y_t + v_cell_h // 2), ("Tracking", y_v + t_cell_h // 2)]:
        txt_img = PILImage.new('RGB', (300, 40), (255, 255, 255))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((0, 0), text, fill=(0, 0, 0), font=font_label)
        txt_bbox = txt_img.getbbox()
        if txt_bbox:
            txt_img = txt_img.crop(txt_bbox)
        txt_img = txt_img.rotate(90, expand=True)
        paste_x = label_w - txt_img.size[0] + 1    # flush right, 4px gap from frames
        paste_y = y_center - txt_img.size[1] // 2 - 80   # 20px upward shift
        canvas.paste(txt_img, (paste_x, paste_y))

    # frame labels with dark background pill for readability
    for col, fi in enumerate(frame_indices):
        x_base = label_w + col * (cell_w + gap)
        frame_text = f"t = {fi}"
        text_bbox = draw.textbbox((0, 0), frame_text, font=font_frame)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad_x, pad_y = 6, 3

        pill_x = x_base + 20
        pill_y = y_t + v_cell_h - text_h - 2 * pad_y - 20
        draw.text((pill_x, pill_y), frame_text,
                  fill=(255, 255, 0), font=font_frame)

        # tracking label (move down)
        pill_y = y_v + t_cell_h - text_h - 2 * pad_y - 20
        draw.text((pill_x, pill_y), frame_text,
                  fill=(255, 255, 0), font=font_frame)

    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)

    canvas.save(output_path, dpi=(200, 200))
    print(f"saved: {output_path}")


def save_side_by_side_gif(tracking_df, video_frames, label, game_time, output_path, fps=3):
    """save a gif with tracking on left and video on right."""
    n_frames = len(tracking_df)

    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{label} at {game_time}", fontsize=14, fontweight='bold')

    def update(fi):
        ax_t.clear()
        ax_v.clear()

        # tracking
        draw_tracking_frame(ax_t, tracking_df.iloc[fi])
        ax_t.set_title("tracking", fontsize=11, fontweight='bold')
        ax_t.text(0.02, 0.02, f"frame {fi}/{n_frames - 1}", transform=ax_t.transAxes,
                  fontsize=8, color='white', verticalalignment='bottom',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # video
        ax_v.imshow(video_frames[fi])
        ax_v.axis('off')
        ax_v.set_title("video", fontsize=11, fontweight='bold')
        ax_v.text(0.02, 0.02, f"frame {fi}/{n_frames - 1}", transform=ax_v.transAxes,
                  fontsize=8, color='white', verticalalignment='bottom',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        return []

    anim = FuncAnimation(fig, update, frames=range(n_frames), blit=False, interval=1000 // fps)
    anim.save(output_path, writer='pillow', fps=fps, dpi=150)
    plt.close(fig)
    print(f"saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="visualize same event from both modalities")
    parser.add_argument('--tracking-dir', default='tracking-dataset')
    parser.add_argument('--video-dir', default='video-dataset')
    parser.add_argument('--split', default='test')
    parser.add_argument('--label', default='GOAL')
    parser.add_argument('--index', type=int, default=0, help='occurrence index for the label')
    parser.add_argument('--output', default=None, help='output path (auto-generated if not set)')
    parser.add_argument('--gif', action='store_true', help='save as gif instead of static figure')
    parser.add_argument('--fps', type=int, default=3, help='fps for gif')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='specific frame indices to show (for static figure)')
    args = parser.parse_args()

    # load annotations from both modalities
    t_ann_path = os.path.join(args.tracking_dir, f"annotations_{args.split}.json")
    v_ann_path = os.path.join(args.video_dir, f"annotations_{args.split}.json")

    with open(t_ann_path) as f:
        t_ann = json.load(f)
    with open(v_ann_path) as f:
        v_ann = json.load(f)

    # find the clip in both
    t_clip_entry = find_clip(t_ann, args.label, args.index)
    v_clip_entry = find_clip(v_ann, args.label, args.index)

    if not t_clip_entry or not v_clip_entry:
        print(f"could not find {args.label} at index {args.index} in {args.split}")
        return

    # verify metadata matches
    t_meta = t_clip_entry['metadata']
    v_meta = v_clip_entry['metadata']

    print(f"tracking: game={t_meta['game_id']} time={t_meta['game_time']} pos={t_meta['position_ms']}ms")
    print(f"video:    game={v_meta['game_id']} time={v_meta['game_time']} pos={v_meta['position_ms']}ms")

    if t_meta['game_id'] != v_meta['game_id'] or t_meta['position_ms'] != v_meta['position_ms']:
        print("WARNING: metadata does not match between modalities!")
    else:
        print("metadata matches across modalities")

    # load the actual clip data
    t_input = t_clip_entry['inputs'][0]
    v_input = v_clip_entry['inputs'][0]

    tracking_df = load_tracking_clip(args.tracking_dir, args.split, t_input['path'])
    video_frames = load_video_clip(args.video_dir, args.split, v_input['path'])

    print(f"tracking clip: {len(tracking_df)} frames")
    print(f"video clip: {video_frames.shape}")

    label_clean = args.label.replace(' ', '_')
    game_time = t_meta['game_time'].replace(' ', '').replace('-', '_').replace(':', '')

    # generate output path
    if args.output:
        output_path = args.output
    else:
        ext = 'gif' if args.gif else 'png'
        output_path = f"modality_comparison_{label_clean}_{game_time}_{args.split}.{ext}"

    if args.gif:
        save_side_by_side_gif(tracking_df, video_frames, args.label, t_meta['game_time'],
                              output_path, fps=args.fps)
    else:
        save_side_by_side_frames(tracking_df, video_frames, args.label, t_meta['game_time'],
                                 output_path, frame_indices=args.frames)


if __name__ == '__main__':
    main()