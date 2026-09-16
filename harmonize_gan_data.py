"""
Harmonizes GAN/diffusion training images through ONE common pipeline:
strips EXIF/ICC metadata, converts to RGB, resizes, and re-encodes as
JPEG at a fixed quality -- identically for real/ and fake/.

Why: the gan_classifier branch hit near-perfect AUC on same-source data
(Roboflow test set, diffusion holdout) but chance-level AUC (0.53) on a
genuinely independent StyleGAN dataset. The likely cause: every "real"
image came from one pipeline (wiki.zip/IMDB-WIKI) and every "fake" image
came from a different pipeline (StyleGAN export / Stable Diffusion
export), each with distinct incidental encoding signatures (ICC profiles,
PNG vs JPEG quirks, compression history) that a CNN can trivially latch
onto instead of learning actual face-level synthesis artifacts. Applying
IDENTICAL re-encoding to every image, real or fake, from every source,
removes that shortcut -- neither class carries a systematic format tell
after this runs.

This does NOT replace training-time augmentation (see
models/gan_classifier.py's build_train_transform) -- it's a one-time
dataset-level fix; augmentation is a per-batch, per-epoch defense against
subtler residual differences. Use both.

Usage:
    python harmonize_gan_data.py --src data/gan_stylegan --dst data/gan_stylegan_harmonized
    python harmonize_gan_data.py --src data/gan_diffusion --dst data/gan_diffusion_harmonized
"""
import argparse
import os
from glob import glob

from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def harmonize_folder(src_dir: str, dst_dir: str, quality: int, max_size: int):
    os.makedirs(dst_dir, exist_ok=True)
    paths = [p for p in glob(os.path.join(src_dir, "*")) if os.path.splitext(p)[1].lower() in IMG_EXTS]

    n_ok, n_fail = 0, 0
    for p in tqdm(paths, desc=f"Harmonizing -> {os.path.basename(dst_dir)}"):
        try:
            img = Image.open(p).convert("RGB")  # drops ICC/EXIF/alpha by re-creating a fresh RGB buffer
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.LANCZOS)
            out_name = os.path.splitext(os.path.basename(p))[0] + ".jpg"
            img.save(os.path.join(dst_dir, out_name), "JPEG", quality=quality)
            n_ok += 1
        except Exception as e:
            n_fail += 1
    print(f"  {os.path.basename(dst_dir)}: {n_ok} ok, {n_fail} failed")
    return n_ok, n_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source dir with real/ and fake/ subfolders")
    parser.add_argument("--dst", required=True, help="Output dir -- will create real/ and fake/")
    parser.add_argument("--quality", type=int, default=92,
                         help="Fixed JPEG quality applied to EVERY image, real and fake alike")
    parser.add_argument("--max_size", type=int, default=512,
                         help="Cap the longer side at this many pixels (keeps files reasonably sized)")
    args = parser.parse_args()

    print(f"[harmonize] {args.src} -> {args.dst} (quality={args.quality}, max_size={args.max_size})")
    real_ok, real_fail = harmonize_folder(os.path.join(args.src, "real"),
                                           os.path.join(args.dst, "real"), args.quality, args.max_size)
    fake_ok, fake_fail = harmonize_folder(os.path.join(args.src, "fake"),
                                           os.path.join(args.dst, "fake"), args.quality, args.max_size)

    print(f"\n[harmonize] done. real={real_ok} ok/{real_fail} failed, fake={fake_ok} ok/{fake_fail} failed")
    print(f"[harmonize] output at {args.dst}")


if __name__ == "__main__":
    main()
