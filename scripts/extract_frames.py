r"""
Extracts aligned face crops from raw FaceForensics++ (FF++) and/or Celeb-DF
videos into the layout train/dataset.py expects:

    data/spatial_train/real/<video_id>_frame_XXXX.jpg     (flat, for spatial + frequency branches)
    data/spatial_train/fake/<video_id>_frame_XXXX.jpg

    data/temporal_clips/real/<video_id>/frame_XXXX.jpg    (per-clip, for the temporal branch)
    data/temporal_clips/fake/<video_id>/frame_XXXX.jpg

Also writes a manifest CSV (video-level) so you can later do a video-level
train/val split instead of a naive frame-level random split — frames from
the same video are highly correlated, so splitting at the frame level (as
the current train/dataset.py does by default) leaks information between
train and val and inflates your validation AUC. Recommended fix once you
have this manifest: filter FrameDataset/ClipDataset by video_id membership
in a train-video-set vs val-video-set rather than random_split() on frames.

--------------------------------------------------------------------------
Expected raw dataset layouts
--------------------------------------------------------------------------

FaceForensics++ (after extracting the zip the download script gives you):
    <ffpp_root>/
      original_sequences/youtube/c23/videos/*.mp4          -> real
      manipulated_sequences/Deepfakes/c23/videos/*.mp4      -> fake
      manipulated_sequences/Face2Face/c23/videos/*.mp4      -> fake
      manipulated_sequences/FaceSwap/c23/videos/*.mp4       -> fake
      manipulated_sequences/NeuralTextures/c23/videos/*.mp4 -> fake

Celeb-DF v2 (after extracting the Google Drive download):
    <celebdf_root>/
      Celeb-real/*.mp4        -> real
      YouTube-real/*.mp4      -> real
      Celeb-synthesis/*.mp4   -> fake

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    python -m scripts.extract_frames --dataset ffpp --input_dir "D:\datasets\FaceForensics++" --compression c23
    python -m scripts.extract_frames --dataset celebdf --input_dir "D:\datasets\Celeb-DF-v2"

Run it once per dataset — it appends into the same data/ output folders, so
you end up with one combined training set across both.
"""

import argparse
import csv
import os
import sys
from glob import glob

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from face_detector import FaceDetector

MANIFEST_PATH = os.path.join(config.DATA_ROOT, "video_manifest.csv")


def discover_ffpp_videos(root: str, compression: str):
    """Returns list of (video_path, label) where label is 'real' or 'fake'."""
    videos = []
    real_glob = os.path.join(root, "original_sequences", "youtube", compression, "videos", "*.mp4")
    for p in glob(real_glob):
        videos.append((p, "real"))

    for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        fake_glob = os.path.join(root, "manipulated_sequences", method, compression, "videos", "*.mp4")
        for p in glob(fake_glob):
            videos.append((p, "fake"))

    return videos


def discover_celebdf_videos(root: str):
    videos = []
    for sub in ["Celeb-real", "YouTube-real"]:
        for p in glob(os.path.join(root, sub, "*.mp4")):
            videos.append((p, "real"))
    for p in glob(os.path.join(root, "Celeb-synthesis", "*.mp4")):
        videos.append((p, "fake"))
    return videos


def extract_from_video(video_path: str, label: str, video_id: str,
                        face_detector: FaceDetector, frames_per_video: int,
                        flat_out_dir: str, clip_out_dir: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] could not open {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    step = max(1, total_frames // frames_per_video)
    target_indices = set(range(0, total_frames, step))

    flat_label_dir = os.path.join(flat_out_dir, label)
    clip_dir = os.path.join(clip_out_dir, label, video_id)
    os.makedirs(flat_label_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_indices:
            faces = face_detector.crop_faces(frame)
            if faces:
                # Take the largest detected face (most likely the primary subject)
                crop, box, conf = max(faces, key=lambda f: (f[1][2]-f[1][0]) * (f[1][3]-f[1][1]))
                fname = f"{video_id}_frame_{frame_idx:05d}.jpg"
                cv2.imwrite(os.path.join(flat_label_dir, fname), crop)
                cv2.imwrite(os.path.join(clip_dir, fname), crop)
                saved += 1
        frame_idx += 1
        if saved >= frames_per_video:
            break

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ffpp", "celebdf"])
    parser.add_argument("--input_dir", required=True, help="Root of the extracted raw dataset")
    parser.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"],
                         help="FF++ only. c23 = standard medium compression (recommended).")
    parser.add_argument("--frames_per_video", type=int, default=20,
                         help="How many face crops to extract per video")
    parser.add_argument("--output_root", default=config.DATA_ROOT)
    args = parser.parse_args()

    flat_out_dir = os.path.join(args.output_root, "spatial_train")
    clip_out_dir = os.path.join(args.output_root, "temporal_clips")
    os.makedirs(flat_out_dir, exist_ok=True)
    os.makedirs(clip_out_dir, exist_ok=True)

    if args.dataset == "ffpp":
        videos = discover_ffpp_videos(args.input_dir, args.compression)
    else:
        videos = discover_celebdf_videos(args.input_dir)

    if not videos:
        print("No videos found — check --input_dir points at the right folder "
              "(see the layout diagram in this script's docstring).")
        return

    print(f"Found {len(videos)} videos ({args.dataset}). Extracting faces...")
    config.apply_runtime_settings()
    face_detector = FaceDetector()

    os.makedirs(config.DATA_ROOT, exist_ok=True)
    manifest_exists = os.path.exists(MANIFEST_PATH)
    manifest_file = open(MANIFEST_PATH, "a", newline="")
    writer = csv.writer(manifest_file)
    if not manifest_exists:
        writer.writerow(["video_id", "source_dataset", "label", "frames_saved", "source_path"])

    total_saved = 0
    for i, (video_path, label) in enumerate(videos):
        video_id = f"{args.dataset}_{os.path.splitext(os.path.basename(video_path))[0]}"
        saved = extract_from_video(
            video_path, label, video_id, face_detector,
            args.frames_per_video, flat_out_dir, clip_out_dir,
        )
        total_saved += saved
        writer.writerow([video_id, args.dataset, label, saved, video_path])
        if (i + 1) % 25 == 0 or (i + 1) == len(videos):
            print(f"  [{i+1}/{len(videos)}] videos processed, {total_saved} face crops saved so far")

    manifest_file.close()
    print(f"\nDone. {total_saved} face crops extracted from {len(videos)} videos.")
    print(f"Flat frames:  {flat_out_dir}")
    print(f"Clip frames:  {clip_out_dir}")
    print(f"Manifest:     {MANIFEST_PATH}  (video-level labels — use this for a leak-free train/val split)")


if __name__ == "__main__":
    main()
