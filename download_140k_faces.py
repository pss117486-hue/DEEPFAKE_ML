"""
Downloads the "140k Real and Fake Faces" dataset (TheKernel01 mirror of
the well-known Kaggle 140k dataset -- real faces from FFHQ/Flickr, fakes
from a separate StyleGAN generation run) and reshapes it into flat
real/fake folders.

This is a GENUINELY INDEPENDENT StyleGAN source from the Roboflow
"deepfake-detection-h3was" dataset gan_stylegan was trained on -- different
collection, different generation batch -- so it's a legitimate holdout
for checking whether the gan_classifier branch actually generalizes to
StyleGAN faces it's never seen, rather than just memorizing the Roboflow
set it trained (and got leaked-evaluated) on.

Requires the `datasets` library:
    pip install datasets

Usage:
    python download_140k_faces.py --dst "C:\\WORK\\URDU_OCR\\deepfake_detector\\data\\gan_stylegan_holdout" --limit 2000
"""
import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

REPO_ID = "TheKernel01/140k-Real-and-Fake-Faces"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", required=True,
                         help="Output root -- will create <dst>/real/ and <dst>/fake/")
    parser.add_argument("--split", default="test",
                         help="Which split to pull (test is the natural holdout choice)")
    parser.add_argument("--limit", type=int, default=2000,
                         help="Max images PER CLASS to save -- this is just for an honest eval "
                              "check, not more training data, so a few thousand is plenty and "
                              "keeps the download/eval fast")
    args = parser.parse_args()

    real_dst = os.path.join(args.dst, "real")
    fake_dst = os.path.join(args.dst, "fake")
    os.makedirs(real_dst, exist_ok=True)
    os.makedirs(fake_dst, exist_ok=True)

    print(f"[download] loading {args.split} split from {REPO_ID} (streaming)...")
    ds = load_dataset(REPO_ID, split=args.split, streaming=True)

    n_real, n_fake = 0, 0
    for i, example in enumerate(tqdm(ds, desc="Saving images")):
        if n_real >= args.limit and n_fake >= args.limit:
            break
        label = example["label"]  # 0 = real, 1 = fake
        img = example["image"]

        if label == 0 and n_real < args.limit:
            img.convert("RGB").save(os.path.join(real_dst, f"140k_{i}.jpg"))
            n_real += 1
        elif label == 1 and n_fake < args.limit:
            img.convert("RGB").save(os.path.join(fake_dst, f"140k_{i}.jpg"))
            n_fake += 1

    print(f"\n[download] done. real={n_real} -> {real_dst}, fake={n_fake} -> {fake_dst}")
    print(f"\nNext -- evaluate the gan branch against this genuinely independent set:")
    print(f'  python -m scripts.eval_image_classification_dataset --root "{args.dst}" '
          f'--real_dirs real --fake_dirs fake')


if __name__ == "__main__":
    main()
