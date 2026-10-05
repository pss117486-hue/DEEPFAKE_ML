"""
Quick sanity/eval check for the pretrained general AI-image detector
(Organika/sdxl-detector) against data you already have on disk with known
ground truth. Not a substitute for a proper independent-source holdout
eval (same lesson as the gan_classifier branch tonight) -- but a fast way
to catch an obviously broken integration or wildly miscalibrated model
before trusting it for anything.

Usage:
    python -m scripts.eval_pretrained_ai_detector \
        --real_dir data/spatial_train/real \
        --fake_dir data/gan_diffusion_harmonized/fake \
        --limit 200
"""
import argparse
import os
import sys
from glob import glob

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.pretrained_ai_detector import predict_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: str, limit: int) -> list:
    paths = [p for p in glob(os.path.join(root, "*")) if os.path.splitext(p)[1].lower() in IMG_EXTS]
    return paths[:limit] if limit else paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--fake_dir", required=True)
    parser.add_argument("--limit", type=int, default=200,
                         help="Max images per class -- this is a quick check, not a full eval")
    args = parser.parse_args()

    config.apply_runtime_settings()

    real_paths = collect_images(args.real_dir, args.limit)
    fake_paths = collect_images(args.fake_dir, args.limit)
    print(f"[eval] {len(real_paths)} real images from {args.real_dir}")
    print(f"[eval] {len(fake_paths)} fake images from {args.fake_dir}")

    y_true, y_score = [], []
    n_errors = 0
    for path in tqdm(real_paths, desc="Scoring real"):
        try:
            result = predict_image(path)
            y_true.append(0)
            y_score.append(result["fake_probability"])
        except Exception as e:
            n_errors += 1
    for path in tqdm(fake_paths, desc="Scoring fake"):
        try:
            result = predict_image(path)
            y_true.append(1)
            y_score.append(result["fake_probability"])
        except Exception as e:
            n_errors += 1

    if n_errors:
        print(f"[eval] {n_errors} images failed to score (corrupt/unreadable), skipped")

    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
        print(f"\n=== AUC: {auc:.4f} (n={len(y_true)}) ===")
    else:
        print("AUC: n/a (only one class scored)")

    n_correct = sum(1 for t, s in zip(y_true, y_score) if (s >= 0.5) == bool(t))
    print(f"Accuracy at 0.5 threshold: {n_correct}/{len(y_true)} ({n_correct/len(y_true):.1%})")

    real_scores = [s for t, s in zip(y_true, y_score) if t == 0]
    fake_scores = [s for t, s in zip(y_true, y_score) if t == 1]
    if real_scores:
        print(f"Real images -- mean fake_probability: {sum(real_scores)/len(real_scores):.3f} "
              f"(want this LOW)")
    if fake_scores:
        print(f"Fake images -- mean fake_probability: {sum(fake_scores)/len(fake_scores):.3f} "
              f"(want this HIGH)")


if __name__ == "__main__":
    main()
