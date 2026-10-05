"""
Diagnostic for the format-shortcut concern: runs ONLY the gan_classifier
branch on a folder of KNOWN-REAL face images that come from a totally
different pipeline than wiki.zip (e.g. your own spatial_train/real/,
which is FF++/Celeb-DF face crops -- unambiguously real, unambiguously
NOT from the same source/encoding pipeline the gan branch's "real" class
was trained on).

Why this matters: a perfect leak-free AUC on the diffusion holdout is
still consistent with the branch having learned "does this image have
wiki.zip's specific encoding/ICC-profile signature" rather than "does
this face have synthesis artifacts" -- because every real image in
training came from wiki.zip and every fake came from a different
pipeline, that confound is present identically in train AND held-out val,
so a leak-free split can't rule it out on its own.

If the branch is exploiting that shortcut, it will likely score a large
fraction of these known-real, different-source images as FAKE (high
gan_score), since they don't match wiki.zip's format signature.
If the branch learned genuine synthesis artifacts, it should score most
of these confidently as real (low gan_score), regardless of source
pipeline/encoding differences.

Usage:
    python -m scripts.eval_gan_known_reals --real_dir data/spatial_train/real --limit 500
"""
import argparse
import os
import sys
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.gan_classifier import load_gan_model, predict_batch as gan_predict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True,
                         help="Folder of KNOWN-REAL images from a different source than the "
                              "gan branch's training data (e.g. data/spatial_train/real)")
    parser.add_argument("--limit", type=int, default=500,
                         help="Max images to sample (this is a diagnostic, not full eval)")
    args = parser.parse_args()

    config.apply_runtime_settings()

    paths = [p for p in glob(os.path.join(args.real_dir, "*")) if os.path.splitext(p)[1].lower() in IMG_EXTS]
    if len(paths) > args.limit:
        import random
        random.seed(42)
        paths = random.sample(paths, args.limit)
    print(f"[diagnostic] sampling {len(paths)} known-real images from {args.real_dir}")

    import cv2
    crops = []
    for p in tqdm(paths, desc="Loading images"):
        img = cv2.imread(p)
        if img is not None:
            crops.append(img)

    model = load_gan_model()
    scores = gan_predict(model, crops)
    scores = np.array(scores)

    n_flagged_fake = int(np.sum(scores >= 0.5))
    print(f"\n=== Results ===")
    print(f"n={len(scores)} known-real images (from a source NOT used in gan branch training)")
    print(f"gan_score distribution: mean={scores.mean():.3f}  median={np.median(scores):.3f}  "
          f"min={scores.min():.3f}  max={scores.max():.3f}")
    print(f"Flagged as FAKE (score >= 0.5): {n_flagged_fake}/{len(scores)} ({n_flagged_fake/len(scores):.1%})")
    print()
    if n_flagged_fake / len(scores) > 0.3:
        print("HIGH false-positive rate on known-real, different-source images.")
        print("This strongly suggests the branch learned a FORMAT/ENCODING shortcut")
        print("(e.g. wiki.zip's specific ICC-profile/PNG-encoding signature) rather than")
        print("genuine synthesis-artifact detection. The 1.0 diffusion-holdout AUC is likely")
        print("inflated by this same confound, not real generalization.")
    else:
        print("Low false-positive rate -- some evidence AGAINST the pure-shortcut theory,")
        print("though this alone doesn't prove genuine artifact learning. Worth testing")
        print("against more known-real sources before fully trusting the branch.")


if __name__ == "__main__":
    main()
