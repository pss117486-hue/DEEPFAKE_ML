"""
Frequency branch: catches GAN/diffusion upsampling artifacts that are
invisible spatially but show up as periodic patterns in the frequency
spectrum (checkerboard artifacts from transposed convolutions, unnatural
high-frequency roll-off from diffusion decoders, etc.)

Pipeline: face crop -> grayscale -> 2D FFT -> log-magnitude spectrum ->
lightweight CNN classifier. Much cheaper than the spatial branch, so batch
size can be pushed higher on 12GB VRAM.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn

import config


def compute_fft_magnitude(gray_img: np.ndarray, size: int = None) -> np.ndarray:
    """gray_img: HxW uint8. Returns normalized log-magnitude spectrum, size x size float32."""
    size = size or config.FREQ_IMG_SIZE
    img = cv2.resize(gray_img, (size, size)).astype(np.float32)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    # normalize to [0, 1]
    magnitude -= magnitude.min()
    denom = magnitude.max() if magnitude.max() > 0 else 1.0
    magnitude /= denom
    return magnitude.astype(np.float32)


class FrequencyClassifier(nn.Module):
    """Small CNN over the FFT magnitude spectrum. Cheap by design."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        feats = self.net(x).flatten(1)
        return self.head(feats).squeeze(-1)


def load_frequency_model(checkpoint_path: str = None, device: str = None) -> FrequencyClassifier:
    checkpoint_path = checkpoint_path or config.FREQ_CHECKPOINT
    device = device or config.DEVICE
    model = FrequencyClassifier()
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        print(f"[frequency] loaded fine-tuned weights from {checkpoint_path}")
    else:
        print(f"[frequency] WARNING: no checkpoint at {checkpoint_path}. "
              f"Randomly initialized — train via train/train_frequency.py before relying on it.")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model: FrequencyClassifier, crops_bgr: list, device: str = None) -> list:
    device = device or config.DEVICE
    specs = []
    for crop in crops_bgr:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        spec = compute_fft_magnitude(gray)
        specs.append(spec)
    if not specs:
        return []
    batch = torch.from_numpy(np.stack(specs)).unsqueeze(1).to(device)  # (N,1,H,W)

    probs_all = []
    bs = config.FREQ_BATCH_SIZE
    for i in range(0, len(batch), bs):
        chunk = batch[i:i + bs]
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             enabled=config.USE_AMP and device == "cuda"):
            logits = model(chunk)
        probs_all.extend(torch.sigmoid(logits).float().cpu().tolist())
    return probs_all
