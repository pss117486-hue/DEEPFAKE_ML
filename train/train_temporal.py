"""
Fine-tune the temporal (CNN+GRU) branch on per-clip sequences. This is the
most VRAM-hungry branch because it back-props through T frames at once —
config.TEMPORAL_BATCH_SIZE is deliberately conservative for 12GB.

Usage:
    python -m train.train_temporal --data_dir data/temporal_clips --epochs 15
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.temporal_branch import TemporalClassifier
from train.dataset import ClipDataset, load_video_split


def make_balanced_clip_sampler(dataset):
    labels = np.array([label for _, label in dataset.clips])
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

    train_ds = ClipDataset(args.data_dir, video_ids=train_ids)
    val_ds = ClipDataset(args.data_dir, video_ids=val_ids)
    print(f"  clips -> train: {len(train_ds)}  val: {len(val_ds)}")

    train_sampler = make_balanced_clip_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=config.TEMPORAL_BATCH_SIZE, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.TEMPORAL_BATCH_SIZE, shuffle=False, num_workers=2)

    model = TemporalClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP and device == "cuda")

    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 enabled=config.USE_AMP and device == "cuda"):
                logits = model(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for clips, labels in val_loader:
                clips = clips.to(device)
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                     enabled=config.USE_AMP and device == "cuda"):
                    logits = model(clips)
                probs = torch.sigmoid(logits).float().cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(labels.numpy())

        auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
        print(f"Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            os.makedirs(os.path.dirname(config.TEMPORAL_CHECKPOINT), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "val_auc": auc}, config.TEMPORAL_CHECKPOINT)
            print(f"  -> saved new best checkpoint (AUC={auc:.4f})")

    print(f"Done. Best val AUC: {best_auc:.4f}. Checkpoint: {config.TEMPORAL_CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
