"""
Spatial branch: frame/crop-level real-vs-fake classifier.

Backbone: EfficientNet-B4 (via timm), which is the workhorse architecture
behind most top FaceForensics++ / DFDC leaderboard solutions and fits
generously in 12GB VRAM at 380px with AMP.

This module defines the model. Weights (config.SPATIAL_CHECKPOINT) must be
either:
  (a) trained by you via train/train_spatial.py on FF++/DFDC/Celeb-DF, or
  (b) a compatible pretrained checkpoint you place at that path.
Without fine-tuned weights this will run on ImageNet-pretrained features
only, which gives a weak but non-random signal — do not trust it standalone
until fine-tuned.
"""

import os
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T

import config


class SpatialClassifier(nn.Module):
    def __init__(self, backbone_name: str = None, pretrained: bool = True):
        super().__init__()
        backbone_name = backbone_name or config.SPATIAL_BACKBONE
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0  # feature extractor
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        logit = self.head(feats)
        return logit.squeeze(-1)  # raw logit; apply sigmoid outside


def build_transform(img_size: int = None):
    img_size = img_size or config.SPATIAL_IMG_SIZE
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_spatial_model(checkpoint_path: str = None, device: str = None) -> SpatialClassifier:
    checkpoint_path = checkpoint_path or config.SPATIAL_CHECKPOINT
    device = device or config.DEVICE

    model = SpatialClassifier(pretrained=(not os.path.exists(checkpoint_path)))
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        print(f"[spatial] loaded fine-tuned weights from {checkpoint_path}")
    else:
        print(f"[spatial] WARNING: no fine-tuned checkpoint at {checkpoint_path}. "
              f"Using ImageNet-only backbone — accuracy will be poor until you "
              f"fine-tune with train/train_spatial.py")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model: SpatialClassifier, crops_bgr: list, device: str = None) -> list:
    """crops_bgr: list of HxWx3 BGR numpy arrays. Returns list of fake-probabilities."""
    import cv2
    device = device or config.DEVICE
    transform = build_transform()
    tensors = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(transform(rgb))
    if not tensors:
        return []
    batch = torch.stack(tensors).to(device)

    probs_all = []
    bs = config.SPATIAL_BATCH_SIZE
    for i in range(0, len(batch), bs):
        chunk = batch[i:i + bs]
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             enabled=config.USE_AMP and device == "cuda"):
            logits = model(chunk)
        probs = torch.sigmoid(logits).float().cpu().tolist()
        probs_all.extend(probs)
    return probs_all
