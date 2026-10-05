"""
GAN/diffusion branch: detects WHOLE-FACE SYNTHESIS (StyleGAN, diffusion-
generated faces) as opposed to the spatial branch's face-SWAP artifacts.

ARCHITECTURE HISTORY (read before retraining):
  v1 (raw RGB input): hit near-perfect same-source AUC (0.99+) but
    chance-level AUC (0.53) on genuinely independent StyleGAN/diffusion
    holdouts.
  v2 (added harmonized preprocessing + training-time augmentation):
    NO improvement (still ~0.53-0.54 on independent holdouts). Ruled out
    pure format/encoding shortcuts -- the model was likely latching onto
    semantic/compositional differences instead (face alignment, framing,
    background composition, pose distribution) that differ systematically
    between StyleGAN export pipelines and don't reflect real generator
    fingerprints.
  v3 (this version): adds a FIXED, non-trainable SRM (Spatial Rich Model)
    high-pass filter bank as the very first step, before the backbone
    sees any pixels. These are classic image-forensics noise-residual
    filters (used widely in manipulation-detection literature, e.g. Zhou
    et al. CVPR 2018) -- each kernel sums to zero, so flat regions,
    alignment, framing, and background are suppressed entirely; only
    high-frequency noise patterns survive. This is the standard fix for
    the failure mode v1/v2 hit (see Wang et al., "CNN-generated images
    are surprisingly easy to detect... for now", CVPR 2020): raw-pixel
    CNNs default to semantic shortcuts unless forced away from them.

INCOMPATIBLE WITH OLDER CHECKPOINTS: this adds new fixed/learnable layers
(SRMFilter, a following BatchNorm) before the backbone. A gan_classifier.pt
saved from v1/v2 CANNOT be loaded into this architecture -- retrain from
scratch via train/train_gan.py. There is no guaranteed payoff here either;
cross-generator GAN/diffusion detection is a genuinely hard, actively
researched problem -- validate against the same independent holdouts
before trusting this version any more than the last one.
"""

import io
import os
import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import config


# Classic SRM (Spatial Rich Model) high-pass kernels from image forensics
# literature. Each is zero-sum, so constant/flat regions produce zero
# response -- only high-frequency noise residuals survive.
_SRM_KERNELS = [
    np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0,
    np.array([[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0,
    np.array([[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,
]


class SRMFilter(nn.Module):
    """
    Fixed, non-trainable noise-residual filter bank. Applies the 3 SRM
    kernels to each of R/G/B independently, then averages across channels
    per filter -- output is a 3-channel noise-residual map, same spatial
    size as the input, so it drops straight into any backbone expecting
    3-channel input. Weights are registered as a buffer (saved with the
    checkpoint for reproducibility, but never updated by the optimizer).
    """

    def __init__(self):
        super().__init__()
        kernels = np.stack(_SRM_KERNELS, axis=0)          # (3, 5, 5)
        weight = torch.from_numpy(kernels).unsqueeze(1)     # (3, 1, 5, 5)
        self.register_buffer("weight", weight)
        self.padding = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.reshape(b * c, 1, h, w)
        out = F.conv2d(x_flat, self.weight, padding=self.padding)   # (b*c, 3, h, w)
        out = out.reshape(b, c, 3, h, w).mean(dim=1)                 # avg over R,G,B -> (b, 3, h, w)
        return out


class GANClassifier(nn.Module):
    def __init__(self, backbone_name: str = None, pretrained: bool = True, use_srm: bool = True):
        super().__init__()
        backbone_name = backbone_name or config.GAN_BACKBONE
        self.use_srm = use_srm
        if use_srm:
            self.srm = SRMFilter()
            # Learnable renormalization -- SRM output has very different
            # scale/statistics than normalized RGB, this lets the backbone
            # receive something in a more familiar range without touching
            # the fixed filter itself.
            self.srm_norm = nn.BatchNorm2d(3, affine=True)

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        if self.use_srm:
            x = self.srm(x)
            x = self.srm_norm(x)
        feats = self.backbone(x)
        logit = self.head(feats)
        return logit.squeeze(-1)


class RandomJPEGCompression:
    """Randomly re-encodes the image through JPEG at a random quality, in
    memory -- still useful even with SRM, since JPEG blocking artifacts
    are themselves a source-specific confound worth randomizing away."""

    def __init__(self, quality_range=(45, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_transform(img_size: int = None):
    """Deterministic, no augmentation -- used at inference AND for the val
    split during training."""
    img_size = img_size or config.GAN_IMG_SIZE
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_train_transform(img_size: int = None):
    """Augmented transform for TRAINING ONLY -- random flip, color jitter,
    mild blur, random JPEG re-compression. Use build_transform() (no
    augmentation) for validation and inference."""
    img_size = img_size or config.GAN_IMG_SIZE
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        RandomJPEGCompression(quality_range=(45, 95), p=0.6),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_gan_model(checkpoint_path: str = None, device: str = None) -> GANClassifier:
    checkpoint_path = checkpoint_path or config.GAN_CHECKPOINT
    device = device or config.DEVICE

    model = GANClassifier(pretrained=(not os.path.exists(checkpoint_path)))
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(state["model_state"] if "model_state" in state else state)
            print(f"[gan] loaded fine-tuned weights from {checkpoint_path}")
        except RuntimeError as e:
            print(f"[gan] ERROR loading {checkpoint_path}: {e}")
            print("[gan] This checkpoint is almost certainly from before the SRM filter was "
                  "added -- the architecture changed and old weights can't be reused. "
                  "Retrain via train/train_gan.py, or delete the old checkpoint.")
            print("[gan] Falling back to a fresh (untrained) model -- scores will be unreliable "
                  "until retrained.")
    else:
        print(f"[gan] WARNING: no fine-tuned checkpoint at {checkpoint_path}. "
              f"Using ImageNet-only backbone -- train via train/train_gan.py first.")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model: GANClassifier, crops_bgr: list, device: str = None) -> list:
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
    bs = config.GAN_BATCH_SIZE
    for i in range(0, len(batch), bs):
        chunk = batch[i:i + bs]
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             enabled=config.USE_AMP and device == "cuda"):
            logits = model(chunk)
        probs = torch.sigmoid(logits).float().cpu().tolist()
        probs_all.extend(probs)
    return probs_all
