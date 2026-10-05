"""
Extracts face crops from the downloaded DFDC chunks (dfdc_train_part_00
through _09) and appends new entries to the existing data/video_manifest.csv
so DFDC videos slot in alongside FF++/Celeb-DF for training.

Expected input layout (matches what you already have):
    data/DFDC_raw/dfdc_train_part_00/dfdc_train_part_0/metadata.json
    data/DFDC_raw/dfdc_train_part_00/dfdc_train_part_0/<video>.mp4
    ... (through part_09)

Output layout (matches FrameDataset / ClipDataset conventions used by the
existing train/dataset.py, same as FF++/Celeb-DF extraction):
    data/dfdc_frames/{real,fake}/dfdc_<chunk>_<video_id>_frame_00001.jpg
    data/dfdc_clips/{real,fake}/dfdc_<chunk>_<video_id>/frame_00001.jpg

video_manifest.csv gets NEW ROWS APPENDED (source_dataset=dfdc) — existing
FF++/Celeb-DF rows are left untouched, so load_video_split() picks up both
seamlessly once you point training at a directory containing both.

Usage:
    python -m scripts.extract_frames_dfdc \
        --dfdc_raw_dir data/DFDC_raw \
        --frames_out data/dfdc_frames \
        --clips_out data/dfdc_clips \
        --manifest data/video_manifest.csv

    # Test on a small subset first:
    python -m scripts.extract_frames_dfdc --max_videos_per_chunk 20
"""
import argparse
import csv
import json
import os
import sys
from glob import glob

import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from face_detector import FaceDetector


def find_chunk_dirs(dfdc_raw_dir):
    """
    Each dfdc_train_part_XX/ contains exactly one subfolder
    (dfdc_train_part_N/) with the videos + metadata.json inside.
    Returns list of (chunk_label, chunk_path) e.g. ("00", ".../dfdc_train_part_0")
    """
    chunks = []
    for part_dir in sorted(glob(os.path.join(dfdc_raw_dir, "dfdc_train_part_*"))):
        chunk_label = os.path.basename(part_dir).replace("dfdc_train_part_", "")
        inner_dirs = [d for d in glob(os.path.join(part_dir, "*")) if os.path.isdir(d)]
        if not inner_dirs:
            continue
        chunks.append((chunk_label, inner_dirs[0]))
    return chunks


def sample_frame_indices(cap, max_frames, sample_fps):
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / sample_fps)))
    indices = list(range(0, total_frames, step))
    return set(indices[:max_frames])


def extract_video(video_path, video_id, label, face_detector, frames_out_dir, clips_out_dir,
                   max_frames, sample_fps):
    """
    Detects faces in sampled frames, saves flat frame crops (for spatial/
    frequency) AND a per-video clip folder (for temporal), tracking only
    the largest/most-confident face per frame for simplicity.
    Returns number of frames successfully saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    indices_to_sample = sample_frame_indices(cap, max_frames, sample_fps)
    label_dir = "real" if label == "real" else "fake"

    flat_dir = os.path.join(frames_out_dir, label_dir)
    clip_dir = os.path.join(clips_out_dir, label_dir, video_id)
    os.makedirs(flat_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices_to_sample:
            faces = face_detector.crop_faces(frame)
            if faces:
                # pick highest-confidence face if multiple detected
                crop, box, conf = max(faces, key=lambda f: f[2])
                fname = f"{video_id}_frame_{saved:05d}.jpg"
                cv2.imwrite(os.path.join(flat_dir, fname), crop)
                cv2.imwrite(os.path.join(clip_dir, f"frame_{saved:05d}.jpg"), crop)
                saved += 1
        frame_idx += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfdc_raw_dir", default=os.path.join(config.DATA_ROOT, "DFDC_raw"))
    parser.add_argument("--frames_out", default=os.path.join(config.DATA_ROOT, "dfdc_frames"))
    parser.add_argument("--clips_out", default=os.path.join(config.DATA_ROOT, "dfdc_clips"))
    parser.add_argument("--manifest", default=os.path.join(config.DATA_ROOT, "video_manifest.csv"))
    parser.add_argument("--max_frames_per_video", type=int, default=config.MAX_FRAMES_PER_VIDEO)
    parser.add_argument("--sample_fps", type=float, default=config.FRAME_SAMPLE_FPS)
    parser.add_argument("--max_videos_per_chunk", type=int, default=None,
                         help="Limit videos processed per chunk (for a quick test run)")
    args = parser.parse_args()

    config.apply_runtime_settings()
    face_detector = FaceDetector()

    chunks = find_chunk_dirs(args.dfdc_raw_dir)
    print(f"[dfdc-extract] found {len(chunks)} chunks: {[c[0] for c in chunks]}")

    manifest_exists = os.path.exists(args.manifest)
    new_rows = []

    for chunk_label, chunk_path in chunks:
        meta_path = os.path.join(chunk_path, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"[dfdc-extract] WARNING: no metadata.json in {chunk_path}, skipping chunk")
            continue
        with open(meta_path) as f:
            metadata = json.load(f)

        video_files = list(metadata.items())
        if args.max_videos_per_chunk:
            video_files = video_files[: args.max_videos_per_chunk]

        for filename, info in tqdm(video_files, desc=f"Chunk {chunk_label}"):
            video_path = os.path.join(chunk_path, filename)
            if not os.path.exists(video_path):
                continue
            label = "real" if info["label"] == "REAL" else "fake"
            base_name = os.path.splitext(filename)[0]
            video_id = f"dfdc_{chunk_label}_{base_name}"

            frames_saved = extract_video(
                video_path, video_id, label, face_detector,
                args.frames_out, args.clips_out,
                args.max_frames_per_video, args.sample_fps,
            )
            new_rows.append({
                "video_id": video_id,
                "source_dataset": "dfdc",
                "label": label,
                "frames_saved": frames_saved,
                "source_path": video_path,
            })

    with open(args.manifest, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "source_dataset", "label",
                                                "frames_saved", "source_path"])
        if not manifest_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    n_real = sum(1 for r in new_rows if r["label"] == "real")
    n_fake = sum(1 for r in new_rows if r["label"] == "fake")
    print(f"[dfdc-extract] appended {len(new_rows)} videos to {args.manifest} "
          f"(real={n_real}, fake={n_fake})")
    print(f"[dfdc-extract] frames written to {args.frames_out}/{{real,fake}}/")
    print(f"[dfdc-extract] clips written to {args.clips_out}/{{real,fake}}/<video_id>/")


if __name__ == "__main__":
    main()
