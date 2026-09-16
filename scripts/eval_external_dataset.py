"""
Batch-evaluates the full video pipeline (VideoDeepfakeDetector: spatial +
frequency + temporal + forensic + fusion) against an EXTERNAL dataset the
model has never been trained on — e.g. SDFVD (Surrey Deepfake Video
Dataset), a completely different research group's data than
FF++/Celeb-DF/DFDC.

This is the most honest generalization check available: unlike
eval_per_dataset.py (which checks held-out videos from datasets the model
WAS trained on), this checks a dataset with zero overlap at all.

Expected input layout:
    <root>/videos_real/*.mp4
    <root>/videos_fake/*.mp4

Usage:
    python -m scripts.eval_external_dataset --root /home/dev/Downloads/SDFVD
"""
import argparse
import os
import sys
from glob import glob

from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from inference.video_inference import VideoDeepfakeDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                         help="Folder containing videos_real/ and videos_fake/ subfolders")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only evaluate the first N videos per class (quick test)")
    parser.add_argument("--video_list", default=None,
                         help="Optional: path to a text file listing exact video paths (one per "
                              "line) to restrict evaluation to -- use this with the holdout list "
                              "from scripts/build_fusion_training_csv.py to test ONLY videos that "
                              "were never used in fusion training, for a leakage-free final check")
    args = parser.parse_args()

    config.apply_runtime_settings()
    detector = VideoDeepfakeDetector()

    real_videos = sorted(glob(os.path.join(args.root, "videos_real", "*")))
    fake_videos = sorted(glob(os.path.join(args.root, "videos_fake", "*")))

    if args.video_list:
        with open(args.video_list) as f:
            allowed = set(line.strip() for line in f if line.strip())
        real_videos = [p for p in real_videos if p in allowed]
        fake_videos = [p for p in fake_videos if p in allowed]
        print(f"[eval-external] restricted to {len(allowed)} videos from {args.video_list}")

    if args.limit:
        real_videos = real_videos[: args.limit]
        fake_videos = fake_videos[: args.limit]

    print(f"[eval-external] {len(real_videos)} real videos, {len(fake_videos)} fake videos")

    all_videos = [(p, 0) for p in real_videos] + [(p, 1) for p in fake_videos]

    results = []
    for path, true_label in tqdm(all_videos, desc="Analyzing"):
        try:
            result = detector.analyze(path)
        except Exception as e:
            print(f"  [skip] {os.path.basename(path)}: {e}")
            continue
        results.append({
            "path": path,
            "true_label": true_label,
            "final_score": result.get("final_score"),
            "verdict": result.get("verdict"),
            "faces_found": result.get("faces_found", result.get("identities_tracked")),
            "spatial_score": result.get("spatial_score"),
            "frequency_score": result.get("frequency_score"),
            "temporal_score": result.get("temporal_score"),
            "forensic_score": result.get("forensic_score"),
            "frames_analyzed": result.get("frames_analyzed"),
        })

    # --- Overall metrics ---
    scored = [r for r in results if r["final_score"] is not None]
    y_true = [r["true_label"] for r in scored]
    y_score = [r["final_score"] for r in scored]

    print(f"\n=== Results on {args.root} ({len(scored)} videos scored) ===\n")

    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
        print(f"AUC: {auc:.4f}")
    else:
        print("AUC: n/a (only one class present in scored results)")

    # Hard verdict accuracy (excluding "uncertain" from correct/incorrect,
    # tallied separately since it's neither a hit nor a miss)
    n_correct, n_wrong, n_uncertain = 0, 0, 0
    for r in scored:
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
    print(f"\nConfident verdicts: {n_decided} ({n_correct} correct, {n_wrong} wrong)"
          f"  Accuracy on confident verdicts: {n_correct / n_decided:.2%}" if n_decided else "")
    print(f"Uncertain verdicts: {n_uncertain} ({n_uncertain/len(scored):.1%} of all videos)")

    # --- Per-branch AUC, if per-branch scores were available ---
    for branch in ["spatial_score", "frequency_score", "temporal_score", "forensic_score"]:
        branch_scored = [r for r in scored if r[branch] is not None]
        if len(branch_scored) < 2:
            continue
        y_true_b = [r["true_label"] for r in branch_scored]
        y_score_b = [r[branch] for r in branch_scored]
        if len(set(y_true_b)) > 1:
            branch_auc = roc_auc_score(y_true_b, y_score_b)
            print(f"  {branch:18s} AUC: {branch_auc:.4f}  (n={len(branch_scored)})")

    # --- Per-video breakdown ---
    print("\n=== Per-video results ===")
    for r in scored:
        true_str = "REAL" if r["true_label"] == 0 else "FAKE"
        mark = "✓" if (
            ("fake" in r["verdict"] and "uncertain" not in r["verdict"] and r["true_label"] == 1) or
            ("real" in r["verdict"] and "uncertain" not in r["verdict"] and r["true_label"] == 0)
        ) else ("?" if "uncertain" in r["verdict"] else "✗")
        print(f"  {mark}  {os.path.basename(r['path']):20s}  true={true_str:4s}  "
              f"frames={r['frames_analyzed']}  "
              f"spatial={r['spatial_score']:.3f}  freq={r['frequency_score']:.3f}  "
              f"temporal={r['temporal_score']:.3f}  forensic={r['forensic_score']:.3f}  "
              f"final={r['final_score']:.3f}  verdict={r['verdict']}")


if __name__ == "__main__":
    main()
