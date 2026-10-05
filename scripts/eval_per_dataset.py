"""
Evaluates the spatial branch checkpoint's AUC SEPARATELY for each
source_dataset present in video_manifest.csv (e.g. ffpp, celebdf, dfdc),
rather than one blended number. This is the key diagnostic for telling
apart two very different situations that look identical in a single
blended val_auc:

  1. Catastrophic forgetting: old-dataset AUC drops after training on new
     data -> the model is genuinely losing what it knew.
  2. Harder val set: old-dataset AUC stays roughly the same, but the new
     dataset's AUC is lower -> the model just hasn't fully learned the new,
     harder distribution yet (expected, not alarming).

Usage:
    python -m scripts.eval_per_dataset \
        --frame_dirs data/spatial_train data/dfdc_frames \
        --checkpoint checkpoints/spatial_b4.pt
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train.dataset import load_video_split
from models.spatial_classifier import load_spatial_model, predict_batch as spatial_predict


def read_manifest(manifest_path):
    """video_id -> (label 0/1, source_dataset)"""
    info = {}
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = 0 if row["label"] == "real" else 1
            info[row["video_id"]] = (label, row.get("source_dataset", "unknown"))
    return info


def collect_val_frames_by_video(frame_dirs, val_ids):
    """
    Scans one or more frame_dir roots (each with real/fake subfolders),
    returns video_id -> list of frame paths, restricted to val_ids.
    """
    by_video = defaultdict(list)
    for frame_dir in frame_dirs:
        for sub in ("real", "fake"):
            for path in glob(os.path.join(frame_dir, sub, "*")):
                vid = os.path.basename(path).rsplit("_frame_", 1)[0]
                if vid in val_ids:
                    by_video[vid].append(path)
    return by_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dirs", nargs="+", required=True,
                         help="One or more frame directories to scan (e.g. data/spatial_train data/dfdc_frames)")
    parser.add_argument("--manifest", default=os.path.join(config.DATA_ROOT, "video_manifest.csv"))
    parser.add_argument("--checkpoint", default=config.SPATIAL_CHECKPOINT)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames_per_video", type=int, default=20)
    args = parser.parse_args()

    config.apply_runtime_settings()
    device = config.DEVICE

    _, val_ids = load_video_split(args.manifest, val_ratio=args.val_ratio, seed=args.seed)
    manifest_info = read_manifest(args.manifest)

    print(f"[eval] loading checkpoint from {args.checkpoint}...")
    model = load_spatial_model(checkpoint_path=args.checkpoint, device=device)

    print("[eval] indexing validation frames across provided directories...")
    frames_by_video = collect_val_frames_by_video(args.frame_dirs, val_ids)
    print(f"[eval] found frame data for {len(frames_by_video)} of {len(val_ids)} val videos")

    # source_dataset -> (list of predicted probs, list of true labels)
    results = defaultdict(lambda: ([], []))
    overall_probs, overall_labels = [], []

    for vid, frame_paths in tqdm(frames_by_video.items(), desc="Evaluating"):
        if vid not in manifest_info:
            continue
        label, source = manifest_info[vid]

        crops = []
        for p in frame_paths[: args.max_frames_per_video]:
            img = cv2.imread(p)
            if img is not None:
                crops.append(img)
        if not crops:
            continue

        scores = spatial_predict(model, crops, device=device)
        video_score = float(np.mean(scores))

        results[source][0].append(video_score)
        results[source][1].append(label)
        overall_probs.append(video_score)
        overall_labels.append(label)

    print("\n=== Per-dataset AUC ===")
    for source in sorted(results.keys()):
        probs, labels = results[source]
        if len(set(labels)) < 2:
            print(f"  {source:15s}: n={len(labels):5d}  AUC=n/a (only one class present)")
            continue
        auc = roc_auc_score(labels, probs)
        n_real = sum(1 for l in labels if l == 0)
        n_fake = sum(1 for l in labels if l == 1)
        print(f"  {source:15s}: n={len(labels):5d} (real={n_real}, fake={n_fake})  AUC={auc:.4f}")

    if len(set(overall_labels)) > 1:
        overall_auc = roc_auc_score(overall_labels, overall_probs)
        print(f"\n  {'OVERALL (blended)':15s}: n={len(overall_labels):5d}  AUC={overall_auc:.4f}")


if __name__ == "__main__":
    main()
