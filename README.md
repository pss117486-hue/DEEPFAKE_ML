# Offline Deepfake Detection System

Fully offline, GPU-accelerated (tuned for 12GB VRAM) deepfake detection for
both images and video. Runs a 4-signal ensemble — spatial artifacts,
frequency-domain artifacts, temporal consistency (video only), and
compression/metadata forensics — then fuses them into a single verdict.

No API keys, no cloud calls, no internet required at inference time once
weights and dependencies are installed.

## Architecture

```
                    ┌─────────────────┐
  image/video  ───► │  Face Detector   │  (MTCNN, or YOLOv8-face)
                    └────────┬─────────┘
                             │ per-face crops
        ┌────────────────────┼────────────────────┬───────────────┐
        ▼                    ▼                     ▼               ▼
  Spatial (B4 CNN)   Frequency (FFT+CNN)   Temporal (CNN+GRU)  Forensics (ELA/metadata)
        │                    │              (video only)            │
        └────────────────────┴──────────┬──────────┴────────────────┘
                                         ▼
                                  Fusion (MLP / weighted avg)
                                         ▼
                              real / fake / uncertain + score
```

## 1. Setup

```bash
cd D:\URDU_OCR\deepfake_detector       # or wherever you place this folder
python -m venv venv
venv\Scripts\activate

# Install the CUDA-matched PyTorch build FIRST — check your driver with `nvidia-smi`
# and get the right index-url from https://pytorch.org/get-started/locally/
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

You'll also need `ffmpeg`/`ffprobe` on PATH for the video forensics module
(container metadata checks) — install via https://ffmpeg.org/download.html
(Windows: grab a build, add the `bin/` folder to PATH).

Verify GPU is picked up:
```bash
python -c "from utils.gpu_utils import print_gpu_status; print_gpu_status()"
```
Should print your GPU name and ~12GB VRAM.

## 2. Important: this ships architecture, not trained weights

The spatial, frequency, and temporal branches will run out of the box on
ImageNet-pretrained backbones, but **that alone gives weak, unreliable
accuracy**. To get to production-grade accuracy you need to fine-tune on
labeled deepfake datasets. This is unavoidable — no offline system can be
accurate without training data, and per the latest research, even
state-of-the-art open-source detectors lose roughly half their accuracy
(AUC) when tested against real-world deepfakes vs. academic benchmarks, so
plan to keep retraining as new generation methods appear.

### Recommended datasets (all require free registration with the dataset authors — not auto-downloadable)
- **FaceForensics++**: https://github.com/ondyari/FaceForensics — best general-purpose starting point, several manipulation methods.
- **DFDC (Deepfake Detection Challenge)**: https://ai.meta.com/datasets/dfdc/ — large scale, diverse identities.
- **Celeb-DF v2**: https://github.com/yuezunli/celeb-deepfakeforensics — higher-quality fakes, good for reducing false negatives on modern generators.
- Consider also collecting a small in-house set of recent diffusion-based
  face-swap/lipsync samples, since the datasets above predate today's
  generation methods and your model will otherwise skew toward detecting
  older-generation artifacts.

### Extracting training frames
Use the face detector to pre-extract aligned face crops from the raw
videos in each dataset into `data/spatial_train/{real,fake}/` (flat) and
`data/temporal_clips/{real,fake}/<clip_id>/` (per-clip, frame order
preserved) — a short extraction script isn't included here since dataset
layouts vary; happy to write one once you've picked which dataset(s)
you're starting with.

### Training
```bash
python -m train.train_spatial   --data_dir data/spatial_train  --epochs 15
python -m train.train_frequency --data_dir data/spatial_train  --epochs 20
python -m train.train_temporal  --data_dir data/temporal_clips --epochs 15

# After the three branches are trained, generate held-out predictions
# (spatial, frequency, temporal, forensic, label) as a CSV, then:
python -m train.train_fusion --csv data/fusion_val_predictions.csv --epochs 200
```

Checkpoints land in `checkpoints/` and are picked up automatically by the
inference scripts.

## 3. Running inference

**Image:**
```bash
python -m inference.image_inference --image path\to\photo.jpg
```

**Video:**
```bash
python -m inference.video_inference --video path\to\clip.mp4
```

Both print a per-branch score breakdown plus a final fused verdict:
`real`, `fake`, or `uncertain — needs human review` (scores between 0.35–0.65
by default, tune via `config.REVIEW_BAND`).

## 4. VRAM tuning notes (12GB card)

| Branch | Default batch | Notes |
|---|---|---|
| Spatial (B4 @ 380px) | 24 | Drop to 16 if you see OOM with other apps open |
| Frequency (lightweight CNN) | 64 | Cheap, can go higher |
| Temporal (CNN+GRU, 16-frame clips) | 4 | Most VRAM-hungry — this is the one to lower first if you hit OOM |

AMP (mixed precision) is on by default (`config.USE_AMP`) — this alone
roughly halves VRAM use for the spatial and temporal branches with
negligible accuracy loss. If you still hit OOM, lower
`config.TEMPORAL_SEQ_LEN` (16 → 8) before touching batch sizes.

## 5. Known limitations / honest caveats

- **Face-only coverage**: this pipeline detects manipulated faces. It does
  not catch full-scene diffusion-generated images/video, background
  inpainting, or non-facial manipulations. If that's in scope, add a
  general image-authenticity branch (e.g. a CNN trained on GAN/diffusion
  vs. camera-real full-frame classification) alongside the face pipeline.
- **Generalization gap**: any detector trained on 2023-era datasets will
  degrade against 2026-era diffusion-based face swaps. Plan to periodically
  refresh training data with recent samples.
- **The `yolov8_face` backend needs its own weights** — your existing
  `yolov8m_UrduDoc.pt` is a document-layout model and will not detect
  faces; either use the default MTCNN backend (no setup needed) or source/
  train a dedicated face-detection YOLOv8 checkpoint.
- Treat the `uncertain` band as a real output state, not an edge case —
  route those to human review rather than forcing a binary call.
