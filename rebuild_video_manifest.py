"""
Rebuilds data/video_manifest.csv from folders that already exist on disk --
the original manifest didn't make it across when this project moved from
D:\\URDU_OCR (a different, earlier machine) to this one, even though the
extracted frame folders did.

This recovers video_id, label, and frames_saved directly from the
extraction folder structure (temporal_clips/{real,fake}/<video_id>/ one
subfolder per video, same for dfdc_clips/{real,fake}/<video_id>/) --
that's everything load_video_split() actually needs for a leak-free
train/val split. source_dataset is inferred from the video_id prefix
convention used by extract_frames.py ("ffpp_"/"celebdf_") and
extract_frames_dfdc.py ("dfdc_<chunk>_").

source_path (the original raw video file) is best-effort: this script
searches your existing raw dataset folders for a matching filename. Videos
where a match isn't found get an empty source_path -- fine for most
scripts (they only need label/frames_saved), but build_fusion_training_csv.py's
IN-DISTRIBUTION portion needs a real source_path to run the full pipeline,
so any row left blank there will just get skipped (harmless, just fewer
in-distribution rows than the original manifest would have given).

Usage:
    python rebuild_video_manifest.py \
        --temporal_dirs data/temporal_clips data/dfdc_clips \
        --raw_search_dirs data/DFDC_raw "data/FaceForensics++" data/raw_celebdf \
        --out data/video_manifest.csv
"""
import argparse
import csv
import os
from glob import glob


def infer_source_dataset(video_id: str) -> str:
    if video_id.startswith("dfdc_"):
        return "dfdc"
    if video_id.startswith("ffpp_"):
        return "ffpp"
    if video_id.startswith("celebdf_"):
        return "celebdf"
    return "unknown"


def scan_clip_dir(clip_root: str) -> list:
    """clip_root/{real,fake}/<video_id>/*.jpg -> list of row dicts."""
    rows = []
    for label_name, label_val in [("real", "real"), ("fake", "fake")]:
        sub = os.path.join(clip_root, label_name)
        if not os.path.isdir(sub):
            continue
        for clip_dir in sorted(glob(os.path.join(sub, "*"))):
            if not os.path.isdir(clip_dir):
                continue
            video_id = os.path.basename(clip_dir)
            frame_count = len(glob(os.path.join(clip_dir, "*.jpg")))
            if frame_count == 0:
                continue
            rows.append({
                "video_id": video_id,
                "source_dataset": infer_source_dataset(video_id),
                "label": label_val,
                "frames_saved": frame_count,
                "source_path": "",  # filled in later if a match is found
            })
    return rows


def build_raw_file_index(raw_search_dirs: list) -> dict:
    """basename (no extension) -> full path, across all raw search dirs."""
    index = {}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for root in raw_search_dirs:
        if not os.path.isdir(root):
            print(f"  [warn] raw search dir not found, skipping: {root}")
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in video_exts:
                    key = os.path.splitext(fname)[0]
                    index[key] = os.path.join(dirpath, fname)
    return index


def try_match_source_path(video_id: str, raw_index: dict) -> str:
    """
    video_id conventions (from extract_frames.py / extract_frames_dfdc.py):
      ffpp_<original_filename>
      celebdf_<original_filename>
      dfdc_<chunk_label>_<original_filename>
    Strip the prefix(es) and look for an exact basename match in the raw index.
    """
    parts = video_id.split("_")
    candidates = []
    if video_id.startswith("dfdc_") and len(parts) >= 3:
        candidates.append("_".join(parts[2:]))       # strip "dfdc_<chunk>_"
    if video_id.startswith("ffpp_"):
        candidates.append(video_id[len("ffpp_"):])
    if video_id.startswith("celebdf_"):
        candidates.append(video_id[len("celebdf_"):])
    candidates.append(video_id)  # fallback: exact video_id as filename

    for cand in candidates:
        if cand in raw_index:
            return raw_index[cand]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal_dirs", nargs="+", required=True,
                         help="Clip-folder roots to scan, e.g. data/temporal_clips data/dfdc_clips")
    parser.add_argument("--raw_search_dirs", nargs="+", default=[],
                         help="Raw dataset roots to search for source_path matches "
                              "(optional but recommended, e.g. data/DFDC_raw)")
    parser.add_argument("--out", default=os.path.join("data", "video_manifest.csv"))
    args = parser.parse_args()

    all_rows = []
    for clip_root in args.temporal_dirs:
        if not os.path.isdir(clip_root):
            print(f"  [warn] temporal dir not found, skipping: {clip_root}")
            continue
        rows = scan_clip_dir(clip_root)
        print(f"[scan] {clip_root}: found {len(rows)} videos")
        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: no videos found in any --temporal_dirs. Check the paths.")
        return

    n_matched = 0
    if args.raw_search_dirs:
        print("[index] building raw-file index (this can take a minute for large folders)...")
        raw_index = build_raw_file_index(args.raw_search_dirs)
        print(f"[index] indexed {len(raw_index)} raw video files")
        for row in all_rows:
            match = try_match_source_path(row["video_id"], raw_index)
            if match:
                row["source_path"] = match
                n_matched += 1
    else:
        print("[index] no --raw_search_dirs given, source_path will be empty for all rows")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "source_dataset", "label", "frames_saved", "source_path"])
        writer.writeheader()
        writer.writerows(all_rows)

    n_real = sum(1 for r in all_rows if r["label"] == "real")
    n_fake = sum(1 for r in all_rows if r["label"] == "fake")
    print(f"\n[done] wrote {len(all_rows)} rows to {args.out}")
    print(f"  real={n_real}, fake={n_fake}")
    print(f"  source_path resolved for {n_matched}/{len(all_rows)} videos "
          f"({len(all_rows) - n_matched} will be skipped by scripts that need real video files, "
          f"e.g. the in-distribution portion of build_fusion_training_csv.py)")

    from collections import Counter
    by_source = Counter(r["source_dataset"] for r in all_rows)
    print(f"  by source_dataset: {dict(by_source)}")


if __name__ == "__main__":
    main()
