"""
Fine-tune the lightweight frequency-domain branch. Much cheaper than the
spatial branch, so this trains fast even on modest data.

Usage:
    python -m train.train_frequency --data_dir data/spatial_train --epochs 20
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.frequency_branch import FrequencyClassifier, compute_fft_magnitude
from train.dataset import FrameDataset, load_video_split
from train.train_spatial import make_balanced_sampler


class FFTDataset(Dataset):
    """Wraps FrameDataset's file list but yields FFT-magnitude tensors instead of RGB crops."""

    def __init__(self, frame_dataset: FrameDataset):
        self.samples = frame_dataset.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        spec = compute_fft_magnitude(gray)
        tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, H, W)
        return tensor, float(label)


def train(args):
    config.apply_runtime_settings()
    device = config.DEVICE

    manifest_path = os.path.join(config.DATA_ROOT, "video_manifest.csv")
    train_ids, val_ids = load_video_split(manifest_path, val_ratio=0.15)

    train_frames = FrameDataset(args.data_dir, video_ids=train_ids)
    val_frames = FrameDataset(args.data_dir, video_ids=val_ids)
    train_ds, val_ds = FFTDataset(train_frames), FFTDataset(val_frames)
    print(f"  frames -> train: {len(train_ds)}  val: {len(val_ds)}")

    train_sampler = make_balanced_sampler(train_frames)
    train_loader = DataLoader(train_ds, batch_size=config.FREQ_BATCH_SIZE, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.FREQ_BATCH_SIZE, shuffle=False, num_workers=4)

    model = FrequencyClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for tensors, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for tensors, labels in val_loader:
                tensors = tensors.to(device)
                probs = torch.sigmoid(model(tensors)).cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(labels.numpy())

        auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
        print(f"Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            os.makedirs(os.path.dirname(config.FREQ_CHECKPOINT), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "val_auc": auc}, config.FREQ_CHECKPOINT)
            print(f"  -> saved new best checkpoint (AUC={auc:.4f})")

    print(f"Done. Best val AUC: {best_auc:.4f}. Checkpoint: {config.FREQ_CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(args)
