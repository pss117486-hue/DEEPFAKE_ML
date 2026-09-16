"""
Reshapes the downloaded Roboflow "Deepfake detection" dataset from its
export layout:
    <src>/train/training_real/*.jpg
    <src>/train/training_fake/*.jpg
    <src>/test/test_real/*.jpg
    <src>/test/test_fake/*.jpg

into the flat layout train/dataset.py's FlatImageDataset expects:
    <dst>/real/*.jpg
    <dst>/fake/*.jpg

Combines train+test into one pool since FlatImageDataset does its own
train/val split internally (train_gan.py) -- Roboflow's own split isn't
meaningful here anyway (see the earlier caveat: this dataset's "train"
vs "test" folders don't necessarily reflect a leakage-free split by the
dataset's original authors, just how the export happened to be organized).

Copies rather than moves, so your original download is left untouched.

Usage:
    python reshape_roboflow_for_gan.py \
        --src "C:\\WORK\\URDU_OCR\\deepfake_detector\\Deepfake-detection-1" \
        --dst "C:\\WORK\\URDU_OCR\\deepfake_detector\\data\\gan_stylegan"
"""
import argparse
import os
import shutil
from glob import glob

from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: str, subdirs: list):
    paths = []
    for sub in subdirs:
        for p in glob(os.path.join(root, sub, "*")):
            if os.path.splitext(p)[1].lower() in IMG_EXTS:
                paths.append(p)
    return paths


def copy_all(src_paths: list, dst_dir: str, prefix: str):
    os.makedirs(dst_dir, exist_ok=True)
    n_copied, n_skipped = 0, 0
    for src_path in tqdm(src_paths, desc=f"Copying into {os.path.basename(dst_dir)}"):
        fname = f"{prefix}_{os.path.basename(src_path)}"
        dst_path = os.path.join(dst_dir, fname)
        if os.path.exists(dst_path):
            n_skipped += 1
            continue
        shutil.copy2(src_path, dst_path)
        n_copied += 1
    return n_copied, n_skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                         help="Root of the downloaded Roboflow dataset (contains train/ and test/)")
    parser.add_argument("--dst", required=True,
                         help="Output root -- will create <dst>/real/ and <dst>/fake/")
    args = parser.parse_args()

    real_dst = os.path.join(args.dst, "real")
    fake_dst = os.path.join(args.dst, "fake")

    real_sources = collect_images(args.src, [
        os.path.join("train", "training_real"),
        os.path.join("test", "test_real"),
    ])
    fake_sources = collect_images(args.src, [
        os.path.join("train", "training_fake"),
        os.path.join("test", "test_fake"),
    ])

    print(f"[reshape] found {len(real_sources)} real images, {len(fake_sources)} fake images in {args.src}")

    n_real_copied, n_real_skipped = copy_all(real_sources, real_dst, "sg")
    n_fake_copied, n_fake_skipped = copy_all(fake_sources, fake_dst, "sg")

    print(f"\n[reshape] done.")
    print(f"  real: {n_real_copied} copied, {n_real_skipped} already existed -> {real_dst}")
    print(f"  fake: {n_fake_copied} copied, {n_fake_skipped} already existed -> {fake_dst}")
    print(f"\nNext:")
    print(f'  python -m train.train_gan --data_dirs "{args.dst}" --epochs 15')


if __name__ == "__main__":
    main()
