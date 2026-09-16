"""
Fine-tune the GAN/diffusion whole-face-synthesis branch.

Unlike the spatial/frequency/temporal branches, this one does NOT use
video_manifest.csv / load_video_split -- GAN/diffusion datasets are
independently-generated still images, not video frames, so there's no
video_id to group by. Split is done at the image level (see
FlatImageDataset in train/dataset.py).

Supports combining MULTIPLE datasets in one run (--data_dirs takes 1+
paths) since real-world GAN/diffusion coverage means mixing generator
families -- e.g. StyleGAN (Yonsei-style datasets), DFFD, and a
diffusion-generated set. Each --data_dirs entry must contain real/ and
fake/ subfolders.

Usage:
    # Single dataset:
    python -m train.train_gan --data_dirs data/gan_train --epochs 15

    # Multiple datasets combined:
    python -m train.train_gan \
        --data_dirs data/stylegan_faces data/dffd data/diffusion_faces \
        --epochs 15
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.gan_classifier import GANClassifier, build_transform, build_train_transform
from train.dataset import FlatImageDataset


def safe_save(state: dict, path: str, max_retries: int = 5, retry_delay: float = 3.0) -> bool:
    """
    Wraps torch.save with retries -- Windows file locks (antivirus scanning
    a freshly-written file, cloud-sync tools like OneDrive, indexing, etc)
    can transiently block a write for a couple seconds. Without this, a
    single lock collision crashes the whole training run and throws away
    that epoch's trained weights, since nothing gets checkpointed.

    Saves to a temp file first, then renames -- avoids ever leaving a
    half-written/corrupt checkpoint if the process is killed mid-save.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)  # atomic on same filesystem
            return True
        except (RuntimeError, OSError, PermissionError) as e:
            last_error = e
            print(f"  [safe_save] attempt {attempt}/{max_retries} failed ({e}); "
                  f"retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    print(f"  [safe_save] FAILED after {max_retries} attempts -- checkpoint NOT saved "
          f"this epoch. Last error: {last_error}")
    print(f"  [safe_save] training continues to next epoch rather than crashing; "
          f"if this keeps happening, check antivirus/cloud-sync locks on {os.path.dirname(path)}")
    return False


def make_balanced_sampler(dataset):
    if len(dataset) == 0:
        raise ValueError(
            "FlatImageDataset produced 0 samples -- check that --data_dirs point at "
            "folders that actually contain real/ and fake/ subfolders with images in them."
        )
    labels = np.array([label for _, label in dataset.samples])
    class_counts = np.bincount(labels.astype(int))
    print(f"  class balance -> real: {class_counts[0]}, fake: {class_counts[1]}")
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]
    return WeightedRandomSampler(
        torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True
    )


def train(args):
    config.apply_runtime_settings()
    device = config.DEVICE

    train_transform = build_train_transform()
    val_transform = build_transform()
    train_ds = FlatImageDataset(args.data_dirs, split="train", val_ratio=args.val_ratio,
                                 seed=args.seed, transform=train_transform)
    val_ds = FlatImageDataset(args.data_dirs, split="val", val_ratio=args.val_ratio,
                               seed=args.seed, transform=val_transform)

    train_sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = GANClassifier(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP and device == "cuda")

    latest_checkpoint_path = config.GAN_CHECKPOINT.replace(".pt", "_latest.pt")
    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 enabled=config.USE_AMP and device == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        scheduler.step()

        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                     enabled=config.USE_AMP and device == "cuda"):
                    logits = model(images)
                probs = torch.sigmoid(logits).float().cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(labels.numpy())

        auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
        print(f"Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  val_auc={auc:.4f}")

        # Always save the latest epoch, regardless of whether it's a new
        # best -- if training gets interrupted later, you still have
        # *something* rather than nothing.
        safe_save({"model_state": model.state_dict(), "val_auc": auc, "epoch": epoch + 1},
                  latest_checkpoint_path)

        if auc > best_auc:
            best_auc = auc
            saved = safe_save({"model_state": model.state_dict(), "val_auc": auc}, config.GAN_CHECKPOINT)
            if saved:
                print(f"  -> saved new best checkpoint (AUC={auc:.4f})")

    print(f"Done. Best val AUC: {best_auc:.4f}. Checkpoint: {config.GAN_CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", required=True,
                         help="One or more dataset roots, each with real/ and fake/ subfolders")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=config.GAN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
