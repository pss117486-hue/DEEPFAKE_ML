"""
Fine-tune the spatial (EfficientNet-B4) branch on your labeled real/fake
frame dataset. Tuned for 12GB VRAM with AMP + gradient accumulation so
effective batch size stays reasonable even at 380px resolution.

Supports RESUMING from the existing checkpoint (spatial_b4.pt) instead of
always starting from the plain ImageNet-pretrained backbone, so training on
a new dataset builds on top of what the model already learned rather than
throwing it away. Use --extra_data_dir to mix in frames from the ORIGINAL
dataset alongside new data — this is the main defense against catastrophic
forgetting (the model overwriting old knowledge while learning new data).

Usage:
    # First time training (no checkpoint exists yet) — same as before:
    python -m train.train_spatial --data_dir data/spatial_train --epochs 15

    # Continuing training on a NEW dataset, resuming from the existing
    # checkpoint, while mixing in some of the original data so old
    # knowledge isn't forgotten:
    python -m train.train_spatial --data_dir data/new_spatial_train \
        --extra_data_dir data/spatial_train --epochs 10

    # Force a fresh run from ImageNet weights (ignore existing checkpoint):
    python -m train.train_spatial --data_dir data/spatial_train --epochs 15 --no_resume
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.spatial_classifier import SpatialClassifier
from train.dataset import FrameDataset, load_video_split


def make_balanced_sampler(dataset):
    """
    Builds a WeightedRandomSampler so each training batch sees roughly equal
    real/fake examples regardless of how imbalanced the raw dataset is.
    Necessary here since FF++/Celeb-DF extraction typically yields far more
    fake frames than real ones (more manipulation methods than pristine videos).
    Works with ConcatDataset too (when --extra_data_dir is used) by pulling
    labels from each underlying dataset's .samples list.
    """
    if isinstance(dataset, ConcatDataset):
        labels = np.array([label for ds in dataset.datasets for _, label in ds.samples])
    else:
        labels = np.array([label for _, label in dataset.samples])
    class_counts = np.bincount(labels)
    print(f"  class balance in this split -> real: {class_counts[0]}, fake: {class_counts[1]}")
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True
    )


def train(args):
    config.apply_runtime_settings()
    device = config.DEVICE

    manifest_path = os.path.join(config.DATA_ROOT, "video_manifest.csv")
    train_ids, val_ids = load_video_split(manifest_path, val_ratio=0.15)

    train_ds = FrameDataset(args.data_dir, video_ids=train_ids)
    val_ds = FrameDataset(args.data_dir, video_ids=val_ids)

    if args.extra_data_dir:
        print(f"  mixing in extra data from {args.extra_data_dir} to help retain prior knowledge")
        extra_train_ds = FrameDataset(args.extra_data_dir, video_ids=train_ids)
        extra_val_ds = FrameDataset(args.extra_data_dir, video_ids=val_ids)
        train_ds = ConcatDataset([train_ds, extra_train_ds])
        val_ds = ConcatDataset([val_ds, extra_val_ds])

    n_train = len(train_ds)
    n_val = len(val_ds)
    print(f"  frames -> train: {n_train}  val: {n_val}")

    train_sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = SpatialClassifier(pretrained=True).to(device)

    # --- Resume from existing checkpoint instead of starting from ImageNet weights ---
    start_best_auc = 0.0
    if args.resume and os.path.exists(config.SPATIAL_CHECKPOINT):
        state = torch.load(config.SPATIAL_CHECKPOINT, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        start_best_auc = state.get("val_auc", 0.0)
        print(f"  [resume] loaded existing weights from {config.SPATIAL_CHECKPOINT} "
              f"(previous val_auc={start_best_auc:.4f})")
    elif args.resume:
        print(f"  [resume] no existing checkpoint found at {config.SPATIAL_CHECKPOINT}, "
              f"starting from ImageNet-pretrained backbone")
    else:
        print("  [resume] --no_resume set, starting from ImageNet-pretrained backbone")

    # Default LR is lower when resuming (gentler fine-tuning of an already-trained
    # model) unless the user explicitly overrides --lr.
    lr = args.lr
    if lr is None:
        lr = 2e-5 if (args.resume and start_best_auc > 0.0) else 1e-4
    print(f"  learning rate: {lr:.2e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP and device == "cuda")

    best_auc = start_best_auc
    accum_steps = max(1, args.effective_batch_size // args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 enabled=config.USE_AMP and device == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels) / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item() * accum_steps
        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # Validation
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
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}  val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            os.makedirs(os.path.dirname(config.SPATIAL_CHECKPOINT), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "val_auc": auc},
                       config.SPATIAL_CHECKPOINT)
            print(f"  -> saved new best checkpoint (AUC={auc:.4f})")

    print(f"Training complete. Best val AUC: {best_auc:.4f}")
    print(f"Checkpoint: {config.SPATIAL_CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                         help="Primary (usually new) dataset directory, real/fake subfolders")
    parser.add_argument("--extra_data_dir", default=None,
                         help="Optional: original/prior dataset directory to mix in, "
                              "so training on new data doesn't overwrite old knowledge")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=config.SPATIAL_BATCH_SIZE)
    parser.add_argument("--effective_batch_size", type=int, default=64,
                         help="Gradient accumulation target; keeps optimization stable "
                              "even though 8-12GB VRAM caps the real batch size lower.")
    parser.add_argument("--lr", type=float, default=None,
                         help="Learning rate. If omitted: 1e-4 for a fresh run, "
                              "2e-5 when resuming from an existing checkpoint.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                         help="Resume from existing checkpoint if present (default: on)")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                         help="Ignore existing checkpoint and start from ImageNet weights")
    args = parser.parse_args()
    train(args)
