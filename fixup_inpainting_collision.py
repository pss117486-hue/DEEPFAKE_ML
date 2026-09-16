"""
Fixes the fake-image collision from download_deepfakeface.py: text2img.zip
and inpainting.zip both reuse the same filenames (0.jpg, 1.jpg, ... --
they're generated from the same source real-image indices), so the
original reshape step silently deduped all 30k inpainting images against
the 30k text2img images already copied in. This copies the missing
inpainting images in with a distinct prefix so both sets actually land.

Requires C:\...\gan_diffusion\_raw\inpainting_extracted to still exist
(i.e. you haven't deleted _raw yet).

Usage:
    python fixup_inpainting_collision.py --dst "C:\\WORK\\URDU_OCR\\deepfake_detector\\data\\gan_diffusion"
"""
import argparse
import os
import shutil

from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_all_images(root: str) -> list:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(dirpath, fname))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", required=True, help="Same --dst you used for download_deepfakeface.py")
    args = parser.parse_args()

    inpainting_dir = os.path.join(args.dst, "_raw", "inpainting_extracted")
    if not os.path.isdir(inpainting_dir):
        print(f"ERROR: {inpainting_dir} not found -- did you already delete _raw? "
              f"If so you'll need to re-run: python download_deepfakeface.py --dst \"{args.dst}\"")
        return

    fake_dst = os.path.join(args.dst, "fake")
    paths = find_all_images(inpainting_dir)
    print(f"[fixup] found {len(paths)} inpainting images to add")

    n_copied, n_skipped = 0, 0
    for src_path in tqdm(paths, desc="Copying inpainting fakes"):
        fname = f"dff_inpaint_{os.path.basename(src_path)}"
        dst_path = os.path.join(fake_dst, fname)
        if os.path.exists(dst_path):
            n_skipped += 1
            continue
        shutil.copy2(src_path, dst_path)
        n_copied += 1

    print(f"[fixup] done. {n_copied} copied, {n_skipped} already existed -> {fake_dst}")

    n_fake_total = len(find_all_images(fake_dst))
    n_real_total = len(find_all_images(os.path.join(args.dst, "real")))
    print(f"[fixup] fake/ now has {n_fake_total} images total, real/ has {n_real_total}")


if __name__ == "__main__":
    main()
