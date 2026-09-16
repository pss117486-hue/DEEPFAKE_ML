r"""
Central configuration for the offline deepfake detection system.
Tuned for a single 8GB VRAM GPU (e.g. RTX 4060 8GB laptop/desktop).
Batch sizes below were reduced from the original 12GB-tuned values.

All paths are relative to the project root by default. Point DATA_ROOT /
CHECKPOINT_ROOT wherever you keep datasets and weights on D:\URDU_OCR.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")          # FF++ / DFDC / Celeb-DF go here
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# NOTE: checkpoints/ on this machine carries a Windows Mandatory Integrity
# Control "High/No-Write" label that silently blocks writes from normal
# (Medium-integrity) processes -- torch.save() fails every time with
# "cannot be opened", regardless of the visible ACL permissions, which look
# fine. Root cause not fully identified (possibly an antivirus/EDR
# auto-quarantine response, given it started right after a rapid write-temp
# -then-rename pattern was introduced). Workaround: the gan_classifier
# checkpoint writes to a SEPARATE folder (gan_checkpoints/, below) that
# doesn't carry the same label. If you fix the label on checkpoints/ itself
# (needs an elevated icacls call — the (OI)(CI) flags need to be quoted
# in cmd.exe/PowerShell), GAN_CHECKPOINT_DIR can be pointed back at
# CHECKPOINT_DIR.
GAN_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "gan_checkpoints")
os.makedirs(GAN_CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Device / precision
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True          # mixed precision — essential to fit 12GB comfortably
CUDNN_BENCHMARK = True  # fixed input sizes -> lets cuDNN autotune, big speedup

# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
FACE_DETECTOR_BACKEND = "mtcnn"     # "mtcnn" (default, no extra weights needed) or "yolov8_face"
YOLOV8_FACE_WEIGHTS = os.path.join(CHECKPOINT_DIR, "yolov8n-face.pt")  # only used if backend=yolov8_face
FACE_CONF_THRESHOLD = 0.90
FACE_MARGIN = 0.35                  # extra crop margin around detected box (fraction of box size)
MIN_FACE_SIZE = 40                  # px, faces smaller than this are skipped

# ---------------------------------------------------------------------------
# Spatial (per-frame CNN) branch
# ---------------------------------------------------------------------------
SPATIAL_BACKBONE = "tf_efficientnet_b4_ns"   # timm model name
SPATIAL_IMG_SIZE = 380                        # native B4 resolution
SPATIAL_BATCH_SIZE = 12                       # reduced for 8GB VRAM (was 24 for 12GB); gradient accumulation via --effective_batch_size keeps effective batch size the same
SPATIAL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "spatial_b4.pt")

# ---------------------------------------------------------------------------
# Frequency (FFT artifact) branch
# ---------------------------------------------------------------------------
FREQ_IMG_SIZE = 256
FREQ_BATCH_SIZE = 32          # reduced for 8GB VRAM (was 64); still lighter model, no AMP used here
FREQ_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "frequency_branch.pt")

# ---------------------------------------------------------------------------
# Temporal (video) branch
# ---------------------------------------------------------------------------
TEMPORAL_SEQ_LEN = 16          # frames per clip fed to the GRU
TEMPORAL_FRAME_STRIDE = 4      # sample every Nth frame from the source video
TEMPORAL_BATCH_SIZE = 2        # reduced for 8GB VRAM (was 4); sequences are the most memory-hungry branch
TEMPORAL_HIDDEN_SIZE = 256
TEMPORAL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "temporal_branch.pt")

# ---------------------------------------------------------------------------
# GAN/diffusion synthesis (whole-face generation) branch
# ---------------------------------------------------------------------------
# Detects whole-face GAN/diffusion synthesis (e.g. StyleGAN, diffusion-
# generated faces) -- a DIFFERENT artifact family from the spatial branch's
# face-swap detection. Same architecture pattern as spatial_classifier.py
# (EfficientNet-B4 + fixed SRM noise-residual front end), separate
# checkpoint, trained via train/train_gan.py on FlatImageDataset (see
# train/dataset.py) since GAN/diffusion datasets are independently
# generated stills, not video frames.
#
# STATUS: DISABLED. Three training attempts (raw RGB; harmonized data +
# augmentation; SRM noise-residual front end) all hit near-perfect
# same-source AUC (0.99+) but only 0.53-0.58 AUC -- essentially chance --
# against genuinely independent StyleGAN/diffusion holdouts. The branch is
# not currently reliable enough to influence real verdicts, so it's
# disabled here rather than silently dragging down fusion output. The
# checkpoint/training code is kept in place for future work (this needs
# substantially more diverse real+fake training sources, not another
# architecture tweak, to actually close the generalization gap). Flip
# GAN_BRANCH_ENABLED back to True once that's been done AND re-validated
# against an independent holdout -- not just a better same-source val_auc.
GAN_BRANCH_ENABLED = False
GAN_BACKBONE = "tf_efficientnet_b4_ns"
GAN_IMG_SIZE = 380
GAN_BATCH_SIZE = 12                            # same VRAM footprint as spatial (same backbone/resolution)
GAN_CHECKPOINT = os.path.join(GAN_CHECKPOINT_DIR, "gan_classifier.pt")  # see note above -- NOT in CHECKPOINT_DIR

# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
FUSION_CHECKPOINT = os.path.join(GAN_CHECKPOINT_DIR, "fusion_head.pt")  # NOT in CHECKPOINT_DIR -- see write-protection note above GAN_CHECKPOINT_DIR
# Fallback weights used only when no learned fusion checkpoint exists at all
# (see train/train_fusion.py). Order: [spatial, frequency, temporal, forensic, gan].
# models/fusion.py now infers the ACTUAL fusion checkpoint's input size from
# its saved weights, so a 4-input (gan disabled, current state) or 5-input
# (gan enabled) checkpoint both load correctly with no code changes needed --
# this array is just the untrained-fallback weighting, unrelated to whichever
# checkpoint size is currently in use.
FUSION_FALLBACK_WEIGHTS = [0.30, 0.15, 0.25, 0.10, 0.20]

# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------
DECISION_THRESHOLD = 0.5        # probability above this => classified as fake
REVIEW_BAND = (0.35, 0.65)      # scores inside this band are flagged "uncertain, needs human review"

# ---------------------------------------------------------------------------
# Video sampling
# ---------------------------------------------------------------------------
MAX_FRAMES_PER_VIDEO = 300      # hard cap so a 2hr video doesn't stall the pipeline
FRAME_SAMPLE_FPS = 5            # analyze at this effective FPS regardless of source FPS


def apply_runtime_settings():
    """Call once at process start."""
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
