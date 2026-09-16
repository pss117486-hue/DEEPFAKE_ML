"""
Temporal branch: video-only. Catches inter-frame inconsistencies that no
single frame reveals — flicker in blending seams, unnatural blink/landmark
jitter, optical-flow discontinuities around the face boundary.

Design: a lightweight CNN backbone (EfficientNet-B0, much cheaper than the
spatial branch's B4 since this runs per-frame across a whole sequence)
extracts per-frame embeddings, then a bidirectional GRU consumes the
sequence and outputs a single clip-level fake-probability.
"""

import os
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T

import config


class TemporalClassifier(nn.Module):
    def __init__(self, hidden_size: int = None):
        super().__init__()
        hidden_size = hidden_size or config.TEMPORAL_HIDDEN_SIZE
        self.backbone = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        feats = self.backbone(x.view(b * t, c, h, w))          # (B*T, feat_dim)
        feats = feats.view(b, t, -1)                             # (B, T, feat_dim)
        _, h_n = self.gru(feats)                                  # h_n: (2, B, hidden)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)               # (B, hidden*2)
        return self.head(h_cat).squeeze(-1)


def build_transform(img_size: int = 224):
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_temporal_model(checkpoint_path: str = None, device: str = None) -> TemporalClassifier:
    checkpoint_path = checkpoint_path or config.TEMPORAL_CHECKPOINT
    device = device or config.DEVICE
    model = TemporalClassifier()
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        print(f"[temporal] loaded fine-tuned weights from {checkpoint_path}")
    else:
        print(f"[temporal] WARNING: no checkpoint at {checkpoint_path}. "
              f"Train via train/train_temporal.py on video clips before relying on it.")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_sequence(model: TemporalClassifier, face_crops_bgr: list, device: str = None) -> float:
    """
    face_crops_bgr: ordered list of face crops (one per sampled frame) for a
    SINGLE tracked identity across the clip. Length should be
    config.TEMPORAL_SEQ_LEN (shorter sequences are looped/padded).
    Returns a single fake-probability for the clip.
    """
    device = device or config.DEVICE
    seq_len = config.TEMPORAL_SEQ_LEN
    if not face_crops_bgr:
        return 0.5  # no signal

    transform = build_transform()
    frames = face_crops_bgr[:seq_len]
    while len(frames) < seq_len:
        frames.append(frames[-1])  # pad by repeating last frame

    tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    clip = torch.stack(tensors).unsqueeze(0).to(device)  # (1, T, C, H, W)

    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                         enabled=config.USE_AMP and device == "cuda"):
        logit = model(clip)
    return float(torch.sigmoid(logit).cpu().item())
