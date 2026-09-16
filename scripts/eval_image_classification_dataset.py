"""
Batch-evaluates the full IMAGE pipeline (face detect + spatial + frequency
+ forensic + fusion) against a flat real/fake classification dataset --
e.g. the Roboflow "Deepfake detection" export
(https://universe.roboflow.com/deepfake-q1kew/deepfake-detection-h3was),
which ships as class-named subfolders rather than the
<video_id>_frame_XXXXX.jpg / video_manifest.csv layout your other eval
scripts expect.

NOTE ON THIS SPECIFIC DATASET: filenames like hard_50_1101.jpg /
mid_295_0111.jpg match the Yonsei "Real and Fake Face Detection" dataset --
GAN-synthesized (StyleGAN-style) whole-face fakes, NOT face-swap
manipulations like FF++/Celeb-DF/DFDC. That makes it a useful but
DIFFERENT generalization check (new artifact family), not a like-for-like
comparison with your SDFVD/DFDC numbers. It's also images only, so the
temporal branch never runs here -- that's expected, not a bug.

Expected input layout (what Roboflow's "folder" export produces):
    <root>/<real_class_name>/*.jpg
    <root>/<fake_class_name>/*.jpg
e.g. --real_dirs test_real --fake_dirs test_fake
     (add training_real / training_fake too if you want to include them,
     but see the note above about that split's provenance)

Usage:
    python -m scripts.eval_image_classification_dataset \
        --root /path/to/deepfake-detection-h3was \
        --real_dirs test_real \
        --fake_dirs test_fake
"""
import argparse
import os
import sys
from glob import glob

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from inference.image_inference import ImageDeepfakeDetector

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: str, subdirs: list):
    paths = []
    for sub in subdirs:
        for p in glob(os.path.join(root, sub, "*")):
            if os.path.splitext(p)[1].lower() in IMG_EXTS:
                paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root folder containing the class subfolders")
    parser.add_argument("--real_dirs", nargs="+", default=["test_real"],
                         help="Subfolder name(s) under --root containing real images")
    parser.add_argument("--fake_dirs", nargs="+", default=["test_fake"],
                         help="Subfolder name(s) under --root containing fake images")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only evaluate the first N images per class (quick test)")
    args = parser.parse_args()

    config.apply_runtime_settings()
    detector = ImageDeepfakeDetector()

    real_paths = collect_images(args.root, args.real_dirs)
    fake_paths = collect_images(args.root, args.fake_dirs)
    if args.limit:
        real_paths = real_paths[: args.limit]
        fake_paths = fake_paths[: args.limit]

    print(f"[eval-images] {len(real_paths)} real images, {len(fake_paths)} fake images "
          f"from {args.root}")

    all_items = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

    results = []
    n_no_face = 0
    for path, true_label in tqdm(all_items, desc="Analyzing"):
        try:
            result = detector.analyze(path)
        except Exception as e:
            print(f"  [skip] {os.path.basename(path)}: {e}")
            continue
        if result.get("faces_found", 0) == 0:
            n_no_face += 1
            continue  # forensic-only score isn't comparable to face-driven verdicts here
        results.append({
            "path": path,
            "true_label": true_label,
            "final_score": result["final_score"],
            "verdict": result["verdict"],
            "spatial_score": max(f["spatial_score"] for f in result["faces"]),
            "frequency_score": max(f["frequency_score"] for f in result["faces"]),
            "gan_score": max(f["gan_score"] for f in result["faces"]),
            "forensic_score": result["faces"][0]["forensic_score"],
        })

    print(f"\n=== Results on {args.root} ({len(results)} images scored, "
          f"{n_no_face} skipped -- no face detected) ===\n")

    if not results:
        print("No images were scored -- check --root and --real_dirs/--fake_dirs point at "
              "folders that actually contain image files.")
        return

    y_true = [r["true_label"] for r in results]
    y_score = [r["final_score"] for r in results]

    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
        print(f"Fused AUC: {auc:.4f}")
    else:
        print("Fused AUC: n/a (only one class present)")

    n_correct, n_wrong, n_uncertain = 0, 0, 0
    for r in results:
        v = r["verdict"]
        predicted_fake = "fake" in v and "uncertain" not in v
        predicted_real = "real" in v and "uncertain" not in v
        if "uncertain" in v:
            n_uncertain += 1
        elif (predicted_fake and r["true_label"] == 1) or (predicted_real and r["true_label"] == 0):
            n_correct += 1
        else:
            n_wrong += 1

    n_decided = n_correct + n_wrong
    if n_decided:
        print(f"Confident verdicts: {n_decided} ({n_correct} correct, {n_wrong} wrong)  "
              f"Accuracy on confident verdicts: {n_correct / n_decided:.2%}")
    if results:
        print(f"Uncertain verdicts: {n_uncertain} ({n_uncertain / len(results):.1%} of scored images)")

    print("\n=== Per-branch AUC ===")
    for branch in ["spatial_score", "frequency_score", "gan_score", "forensic_score"]:
        y_true_b = [r["true_label"] for r in results]
        y_score_b = [r[branch] for r in results]
        if len(set(y_true_b)) > 1:
            branch_auc = roc_auc_score(y_true_b, y_score_b)
            print(f"  {branch:18s} AUC: {branch_auc:.4f}")


if __name__ == "__main__":
    main()
