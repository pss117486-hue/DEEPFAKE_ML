"""
Trains the small fusion MLP on top of the branch scores present in the
input CSV. Requires a CSV of held-out predictions (NOT the same data the
branches were trained on, or fusion will overfit to correlated errors).

Expected CSV columns: spatial,frequency,temporal,forensic[,gan],label
(label: 0=real, 1=fake). The gan column is OPTIONAL -- while
config.GAN_BRANCH_ENABLED is False (see config.py for why), it's normal
and expected for the CSV to have only 4 feature columns; the model size
is inferred from whatever columns are actually present, so this works
correctly either way without needing to hardcode which branches exist.

Usage:
    python -m train.train_fusion --csv data/fusion_val_predictions.csv --epochs 200
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.fusion import FusionMLP

ALL_POSSIBLE_FEATURES = ["spatial", "frequency", "temporal", "forensic", "gan"]


def train(args):
    device = config.DEVICE
    df = pd.read_csv(args.csv)

    feature_cols = [c for c in ALL_POSSIBLE_FEATURES if c in df.columns]
    print(f"[train_fusion] using feature columns: {feature_cols}")
    if "gan" not in feature_cols:
        print("[train_fusion] no 'gan' column in this CSV -- training a "
              f"{len(feature_cols)}-input fusion head (expected while the gan "
              "branch is disabled; see config.GAN_BRANCH_ENABLED).")

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)

    # n_inputs matches whatever columns are ACTUALLY in this CSV -- avoids
    # the mismatch you'd get from FusionMLP()'s default not lining up with
    # a 4-column (gan-disabled) or 5-column (gan-enabled) dataset.
    model = FusionMLP(n_inputs=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
            auc = roc_auc_score(y_val, val_probs)
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}  val_auc={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                os.makedirs(os.path.dirname(config.FUSION_CHECKPOINT), exist_ok=True)
                torch.save({"model_state": model.state_dict(), "val_auc": auc,
                            "feature_cols": feature_cols}, config.FUSION_CHECKPOINT)

    print(f"Done. Best val AUC: {best_auc:.4f}. Checkpoint: {config.FUSION_CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
