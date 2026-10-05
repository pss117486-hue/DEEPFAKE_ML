"""
Generates data/fusion_val_predictions.csv by running the three trained
branch checkpoints (spatial, frequency, temporal) plus the forensic
scorer over the VALIDATION-only videos (per load_video_split's leak-free
split, same seed/ratio as branch training), producing one row per
validation video:

    spatial,frequency,temporal,forensic,label

- spatial/frequency: averaged over that video's held-out frame crops
- temporal: single clip-level score from that video's clip folder
- forensic: ELA/metadata score computed on one representative frame
- label: 0=real, 1=fake (from video_manifest.csv)

This must run on validation videos only, never train videos, or the
fusion head will overfit to the same correlated errors the branches
already memorized on train data.

Usage:
    python -m scripts.generate_fusion_predictions \
        --manifest data/video_manifest.csv \
        --spatial_dir data/spatial_train \
        --temporal_dir data/temporal_clips \
        --out data/fusion_val_predictions.csv
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from train.dataset import load_video_split
from models.spatial_classifier import load_spatial_model, predict_batch as spatial_predict
from models.frequency_branch import load_frequency_model, predict_batch as freq_predict
from models.temporal_branch import load_temporal_model, predict_sequence as temporal_predict
from forensics.metadata_ela import forensic_score


def read_manifest_labels(manifest_path):
    """video_id -> 0/1 label"""
    labels = {}
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["video_id"]] = 0 if row["label"] == "real" else 1
    return labels


def collect_val_frames_by_video(spatial_dir, val_ids):
    """
    spatial_dir/{real,fake}/<video_id>_frame_XXXXX.jpg
    Returns: video_id -> list of frame file paths (val videos only)
    """
    by_video = defaultdict(list)
    for sub in ("real", "fake"):
        for path in glob(os.path.join(spatial_dir, sub, "*")):
            vid = os.path.basename(path).rsplit("_frame_", 1)[0]
            if vid in val_ids:
                by_video[vid].append(path)
    return by_video


def collect_val_clip_dirs(temporal_dir, val_ids):
    """
    temporal_dir/{real,fake}/<video_id>/frame_XXXXX.jpg
    Returns: video_id -> clip_dir path (val videos only)
    """
    clip_dirs = {}
    for sub in ("real", "fake"):
        for clip_dir in glob(os.path.join(temporal_dir, sub, "*")):
            vid = os.path.basename(clip_dir)
            if vid in val_ids:
                clip_dirs[vid] = clip_dir
    return clip_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(config.DATA_ROOT, "video_manifest.csv"))
    parser.add_argument("--spatial_dir", default=os.path.join(config.DATA_ROOT, "spatial_train"))
    parser.add_argument("--temporal_dir", default=os.path.join(config.DATA_ROOT, "temporal_clips"))
    parser.add_argument("--out", default=os.path.join(config.DATA_ROOT, "fusion_val_predictions.csv"))
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames_per_video", type=int, default=20,
                         help="cap frames averaged per video for spatial/frequency (speed)")
    args = parser.parse_args()

    config.apply_runtime_settings()
    device = config.DEVICE

    # Same split used during branch training -> val_ids are held-out videos
    _, val_ids = load_video_split(args.manifest, val_ratio=args.val_ratio, seed=args.seed)
    labels = read_manifest_labels(args.manifest)

    print("[fusion-gen] loading trained branch checkpoints...")
    spatial_model = load_spatial_model()
    freq_model = load_frequency_model()
    temporal_model = load_temporal_model()

    print("[fusion-gen] indexing validation frames/clips on disk...")
    frames_by_video = collect_val_frames_by_video(args.spatial_dir, val_ids)
    clip_dirs_by_video = collect_val_clip_dirs(args.temporal_dir, val_ids)

    # Only videos present in both spatial frames AND temporal clips can get
    # a full row (all three model scores). Videos missing one are skipped.
    common_ids = sorted(set(frames_by_video) & set(clip_dirs_by_video))
    skipped = sorted(set(val_ids) - set(common_ids))
    print(f"[fusion-gen] {len(common_ids)} val videos have both frame crops and clip data "
          f"({len(skipped)} val videos skipped due to missing data)")

    rows = []
    for vid in tqdm(common_ids, desc="Generating fusion predictions"):
        frame_paths = frames_by_video[vid][: args.max_frames_per_video]
        crops = []
        for p in frame_paths:
            img = cv2.imread(p)
            if img is not None:
                crops.append(img)
        if not crops:
            continue

        spatial_scores = spatial_predict(spatial_model, crops, device=device)
        freq_scores = freq_predict(freq_model, crops, device=device)
        spatial_avg = float(np.mean(spatial_scores))
        freq_avg = float(np.mean(freq_scores))

        clip_dir = clip_dirs_by_video[vid]
        clip_frame_paths = sorted(glob(os.path.join(clip_dir, "*.jpg")))
        clip_crops = [cv2.imread(p) for p in clip_frame_paths]
        clip_crops = [c for c in clip_crops if c is not None]
        temporal_score = temporal_predict(temporal_model, clip_crops, device=device)

        # forensic score computed on one representative frame for this video
        forensic = forensic_score(frame_paths[0], is_video=False)

        rows.append({
            "spatial": spatial_avg,
            "frequency": freq_avg,
            "temporal": temporal_score,
            "forensic": forensic,
            "label": labels[vid],
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["spatial", "frequency", "temporal", "forensic", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[fusion-gen] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
