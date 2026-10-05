"""
Builds a combined fusion-training CSV using:
  1. The existing in-distribution validation videos (FF++/Celeb-DF/DFDC),
     scored via the FULL video pipeline (VideoDeepfakeDetector.analyze),
     not just frame-averaging -- keeps aggregation consistent with how
     the model is actually used at inference time.
  2. Roughly half of the external SDFVD dataset (balanced real/fake),
     scored the same way.

The REMAINING half of SDFVD is written to a holdout list file and never
touched here -- it stays available for an honest, leakage-free final test
via scripts/eval_external_dataset.py --video_list <holdout file>.

Rationale: the original fusion head was trained only on in-distribution
predictions, where temporal happened to be very informative and frequency
wasn't -- so it learned to overweight temporal and underweight frequency.
That weighting doesn't transfer to genuinely new data (frequency actually
generalized best in our SDFVD test, temporal was pure noise). Mixing in
real out-of-distribution examples during fusion training is the fix.

Usage:
    python -m scripts.build_fusion_training_csv \
        --manifest data/video_manifest.csv \
        --external_root /home/dev/Downloads/SDFVD \
        --out data/fusion_val_predictions_v2.csv \
        --holdout_list data/sdfvd_holdout_videos.txt
"""
import argparse
import csv
import os
import random
import sys
from glob import glob

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train.dataset import load_video_split
from inference.video_inference import VideoDeepfakeDetector


def resolve_source_path(source_path: str) -> str:
    """
    video_manifest.csv's source_path may be a Windows-style absolute path
    from the original machine (e.g. D:\\URDU_OCR\\deepfake_detector\\data\\...)
    or a Linux path from this machine (e.g. .../deepfake_detector/data/DFDC_raw/...).
    Both cases contain a 'data' path segment -- reconstruct relative to this
    machine's actual config.DATA_ROOT rather than trusting the stored
    absolute path directly.
    """
    parts = source_path.replace("\\", "/").split("/")
    if "data" in parts:
        idx = parts.index("data")
        rel_parts = parts[idx + 1:]
        return os.path.join(config.DATA_ROOT, *rel_parts)
    return source_path


def score_video(detector, path: str, label: int):
    try:
        result = detector.analyze(path)
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return None
    if result.get("spatial_score") is None:
        return None  # e.g. "no face detected" case
    return {
        "spatial": result["spatial_score"],
        "frequency": result["frequency_score"],
        "temporal": result["temporal_score"] if result["temporal_score"] is not None else 0.5,
        "forensic": result["forensic_score"],
        "label": label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(config.DATA_ROOT, "video_manifest.csv"))
    parser.add_argument("--external_root", required=True,
                         help="Folder with videos_real/ and videos_fake/ subfolders (e.g. SDFVD)")
    parser.add_argument("--out", default=os.path.join(config.DATA_ROOT, "fusion_val_predictions_v2.csv"))
    parser.add_argument("--holdout_list", default=os.path.join(config.DATA_ROOT, "sdfvd_holdout_videos.txt"))
    parser.add_argument("--external_train_frac", type=float, default=0.5,
                         help="Fraction of external dataset to use for fusion training; "
                              "rest is held out for a final honest test")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_in_dist_videos", type=int, default=500,
                         help="Cap on in-distribution val videos to score (full video "
                              "pipeline is slower than frame-only scoring; 500 is plenty "
                              "for fusion training and keeps this step fast)")
    args = parser.parse_args()

    config.apply_runtime_settings()
    detector = VideoDeepfakeDetector()
    rng = random.Random(args.seed)

    rows = []

    # --- Part 1: in-distribution validation videos ---
    print("[build-fusion] loading in-distribution val split...")
    _, val_ids = load_video_split(args.manifest, val_ratio=args.val_ratio, seed=args.seed)

    manifest_rows = {}
    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_rows[row["video_id"]] = row

    val_id_list = list(val_ids)
    rng.shuffle(val_id_list)
    val_id_list = val_id_list[: args.max_in_dist_videos]

    print(f"[build-fusion] scoring {len(val_id_list)} in-distribution val videos via full pipeline...")
    for vid in tqdm(val_id_list, desc="In-distribution"):
        row = manifest_rows.get(vid)
        if row is None or int(row.get("frames_saved", 0)) == 0:
            continue
        path = resolve_source_path(row["source_path"])
        if not os.path.exists(path):
            continue
        label = 0 if row["label"] == "real" else 1
        scored = score_video(detector, path, label)
        if scored:
            rows.append(scored)

    print(f"[build-fusion] {len(rows)} in-distribution rows collected so far")

    # --- Part 2: external dataset (SDFVD), split train/holdout ---
    real_videos = sorted(glob(os.path.join(args.external_root, "videos_real", "*")))
    fake_videos = sorted(glob(os.path.join(args.external_root, "videos_fake", "*")))
    rng.shuffle(real_videos)
    rng.shuffle(fake_videos)

    n_real_train = int(len(real_videos) * args.external_train_frac)
    n_fake_train = int(len(fake_videos) * args.external_train_frac)

    real_train, real_holdout = real_videos[:n_real_train], real_videos[n_real_train:]
    fake_train, fake_holdout = fake_videos[:n_fake_train], fake_videos[n_fake_train:]

    print(f"[build-fusion] external dataset: {len(real_train)} real + {len(fake_train)} fake "
          f"for fusion training; {len(real_holdout)} real + {len(fake_holdout)} fake held out")

    external_train = [(p, 0) for p in real_train] + [(p, 1) for p in fake_train]
    for path, label in tqdm(external_train, desc="External (training portion)"):
        scored = score_video(detector, path, label)
        if scored:
            rows.append(scored)

    # Write the holdout list so eval_external_dataset.py can restrict to exactly these files
    with open(args.holdout_list, "w") as f:
        for p in real_holdout + fake_holdout:
            f.write(p + "\n")
    print(f"[build-fusion] wrote {len(real_holdout) + len(fake_holdout)} holdout video paths "
          f"to {args.holdout_list}")

    # --- Write combined fusion-training CSV ---
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["spatial", "frequency", "temporal", "forensic", "label"])
        writer.writeheader()
        writer.writerows(rows)

    n_real = sum(1 for r in rows if r["label"] == 0)
    n_fake = sum(1 for r in rows if r["label"] == 1)
    print(f"[build-fusion] wrote {len(rows)} total rows to {args.out} (real={n_real}, fake={n_fake})")
    print(f"[build-fusion] next: python -m train.train_fusion --csv {args.out} --epochs 200")


if __name__ == "__main__":
    main()
