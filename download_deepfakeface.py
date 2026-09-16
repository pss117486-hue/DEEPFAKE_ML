"""
Downloads the OpenRL/DeepFakeFace dataset from Hugging Face
(https://huggingface.co/datasets/OpenRL/DeepFakeFace) and reshapes it into
the flat real/fake layout FlatImageDataset expects.

Deliberately downloads only:
    - wiki.zip        (real images, from IMDB-WIKI)
    - text2img.zip     (fake -- Stable Diffusion 1.5 text-to-image)
    - inpainting.zip   (fake -- Stable Diffusion Inpainting)

and SKIPS insight.zip (fake -- generated via the InsightFace face-SWAP
toolbox, not diffusion). That subset is a different artifact family
(same category as your existing FF++/DFDC/Celeb-DF swap data) and mixing
it in would dilute the diffusion-specific signal this branch needs --
your existing spatial/frequency branches already cover swap detection.

~4GB total download (three zips). Requires huggingface_hub (already
installed on this machine as a dependency of timm/roboflow).

Usage:
    python download_deepfakeface.py --dst "C:\\WORK\\URDU_OCR\\deepfake_detector\\data\\gan_diffusion"
"""
import argparse
import os
import shutil
import zipfile
from glob import glob

from huggingface_hub import hf_hub_download
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPO_ID = "OpenRL/DeepFakeFace"


def download_and_extract(filename: str, work_dir: str) -> str:
    print(f"[download] fetching {filename} from {REPO_ID}...")
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset",
                                local_dir=work_dir)
    extract_dir = os.path.join(work_dir, filename.replace(".zip", "_extracted"))
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"  already extracted at {extract_dir}, skipping")
        return extract_dir
    print(f"[extract] {filename} -> {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def find_all_images(root: str) -> list:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(dirpath, fname))
    return paths


def copy_all(src_paths: list, dst_dir: str, prefix: str):
    os.makedirs(dst_dir, exist_ok=True)
    n_copied, n_skipped = 0, 0
    for src_path in tqdm(src_paths, desc=f"Copying into {os.path.basename(dst_dir)} ({prefix})"):
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
    parser.add_argument("--dst", required=True,
                         help="Output root -- will create <dst>/real/ and <dst>/fake/")
    parser.add_argument("--work_dir", default=None,
                         help="Where to download/extract zips before reshaping "
                              "(defaults to <dst>/_raw)")
    args = parser.parse_args()

    work_dir = args.work_dir or os.path.join(args.dst, "_raw")
    os.makedirs(work_dir, exist_ok=True)

    real_dst = os.path.join(args.dst, "real")
    fake_dst = os.path.join(args.dst, "fake")

    wiki_dir = download_and_extract("wiki.zip", work_dir)
    text2img_dir = download_and_extract("text2img.zip", work_dir)
    inpainting_dir = download_and_extract("inpainting.zip", work_dir)

    real_paths = find_all_images(wiki_dir)
    fake_paths = find_all_images(text2img_dir) + find_all_images(inpainting_dir)

    print(f"[reshape] found {len(real_paths)} real images, {len(fake_paths)} fake images")

    n_real_copied, n_real_skipped = copy_all(real_paths, real_dst, "dff")
    n_fake_copied, n_fake_skipped = copy_all(fake_paths, fake_dst, "dff")

    print(f"\n[reshape] done.")
    print(f"  real: {n_real_copied} copied, {n_real_skipped} already existed -> {real_dst}")
    print(f"  fake: {n_fake_copied} copied, {n_fake_skipped} already existed -> {fake_dst}")
    print(f"\nYou can now delete {work_dir} to free disk space (raw zips + extracted copies "
          f"aren't needed once the reshaped real/fake folders exist).")
    print(f"\nNext -- train on BOTH GAN datasets together:")
    print(f'  python -m train.train_gan --data_dirs "{args.dst}" "<path to gan_stylegan>" --epochs 15')


if __name__ == "__main__":
    main()
