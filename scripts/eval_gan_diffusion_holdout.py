"""
Isolates a genuinely leak-free read on diffusion-generalization from the
gan_classifier checkpoint. Unlike the stylegan data (which had train/test
combined before training, so any post-hoc "test" eval is contaminated),
gan_diffusion's split happened INSIDE train_gan.py's FlatImageDataset --
same seed, same val_ratio -- so re-running that exact split in isolation
recovers the set of diffusion images the model never trained on.

This only works if you pass the SAME seed/val_ratio/data_dirs the real
training run used (defaults here match train_gan.py's own defaults).

Usage:
    python -m scripts.eval_gan_diffusion_holdout --data_dir data/gan_diffusion
"""
import argparse
import os
import sys

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.gan_classifier import load_gan_model, build_transform
from train.dataset import FlatImageDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                         help="The gan_diffusion folder used in training (real/ and fake/ subfolders)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                         help="MUST match what train_gan.py used (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                         help="MUST match what train_gan.py used (default: 42)")
    args = parser.parse_args()

    config.apply_runtime_settings()
    device = config.DEVICE

    transform = build_transform()
    # split="val" on the SAME data_dir/seed/val_ratio recovers exactly the
    # images that were held out during the real training run -- these were
    # never seen in gradient updates, regardless of what the combined
    # stylegan+diffusion training run's blended val_auc showed.
    val_ds = FlatImageDataset([args.data_dir], split="val", val_ratio=args.val_ratio,
                               seed=args.seed, transform=transform)

    if len(val_ds) == 0:
        print("ERROR: 0 samples in this val split -- check --data_dir, --val_ratio, --seed "
              "match what train_gan.py actually used.")
        return

    val_loader = DataLoader(val_ds, batch_size=config.GAN_BATCH_SIZE, shuffle=False, num_workers=4)

    model = load_gan_model()

    val_probs, val_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating diffusion holdout"):
            images = images.to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 enabled=config.USE_AMP and device == "cuda"):
                logits = model(images)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            val_probs.extend(probs)
            val_labels.extend(labels.numpy())

    auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
    n_real = sum(1 for l in val_labels if l == 0)
    n_fake = sum(1 for l in val_labels if l == 1)
    print(f"\n=== Diffusion holdout AUC: {auc:.4f} (n={len(val_labels)}, real={n_real}, fake={n_fake}) ===")
    print("This is a genuinely leak-free number IF --val_ratio/--seed match the real training run.")


if __name__ == "__main__":
    main()
