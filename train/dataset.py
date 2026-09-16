"""
Dataset loaders for fine-tuning.

Expected directory layout (standard for FF++ / DFDC / Celeb-DF after you've
extracted faces from the raw videos):

    data/
      real/
        video1_frame0001.jpg
        video1_frame0002.jpg
        ...
      fake/
        video2_frame0001.jpg
        ...

For the temporal branch, instead organize as one subfolder per clip so
frame order/identity is preserved:

    data/
      real/
        clip_001/frame_0001.jpg, frame_0002.jpg, ...
      fake/
        clip_002/frame_0001.jpg, ...

Note on datasets: FaceForensics++, DFDC, and Celeb-DF all require you to
request access / accept a usage agreement directly from the dataset
authors — they are not auto-downloadable. See README.md for links.
"""

import csv
import os
import random
from glob import glob

import cv2
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spatial_classifier import build_transform as spatial_transform
from models.temporal_branch import build_transform as temporal_transform
import config


def load_video_split(manifest_path: str, val_ratio: float = 0.15, seed: int = 42):
    """
    Reads data/video_manifest.csv (written by scripts/extract_frames.py) and
    splits at the VIDEO level, stratified by label, so frames/clips from the
    same source video never appear in both train and val. Splitting at the
    frame level instead (e.g. via random_split on a flat file list) leaks
    information between train/val since frames from one video are highly
    correlated, and inflates validation AUC without reflecting real
    generalization.

    Returns (train_video_ids, val_video_ids) as sets of strings.
    """
    real_ids, fake_ids = [], []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["frames_saved"]) == 0:
                continue
            (real_ids if row["label"] == "real" else fake_ids).append(row["video_id"])

    rng = random.Random(seed)
    rng.shuffle(real_ids)
    rng.shuffle(fake_ids)

    def split(ids):
        n_val = max(1, int(len(ids) * val_ratio))
        return set(ids[n_val:]), set(ids[:n_val])

    real_train, real_val = split(real_ids)
    fake_train, fake_val = split(fake_ids)
    train_ids = real_train | fake_train
    val_ids = real_val | fake_val
    print(f"[split] videos -> train: {len(train_ids)} (real={len(real_train)}, fake={len(fake_train)})  "
          f"val: {len(val_ids)} (real={len(real_val)}, fake={len(fake_val)})")
    return train_ids, val_ids


class FrameDataset(Dataset):
    """
    Flat real/fake frame dataset for the spatial and frequency branches.

    If video_ids is given, only frames whose video_id (parsed from the
    filename, matching the "<video_id>_frame_XXXXX.jpg" convention used by
    scripts/extract_frames.py) is in that set are included — use this with
    load_video_split() to build leak-free train/val datasets.
    """

    def __init__(self, root_dir: str, img_size: int = None, transform=None, video_ids: set = None):
        self.samples = []
        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")
        for label, sub_dir in [(0, real_dir), (1, fake_dir)]:
            for path in glob(os.path.join(sub_dir, "*")):
                if video_ids is not None:
                    vid = os.path.basename(path).rsplit("_frame_", 1)[0]
                    if vid not in video_ids:
                        continue
                self.samples.append((path, label))
        random.shuffle(self.samples)

        self.transform = transform or spatial_transform(img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        return tensor, float(label)


class ClipDataset(Dataset):
    """
    Per-clip sequence dataset for the temporal branch.

    If video_ids is given, only clips whose folder name is in that set are
    included (clip folders are named after video_id by extract_frames.py) —
    use with load_video_split() for a leak-free split.
    """

    def __init__(self, root_dir: str, seq_len: int = None, video_ids: set = None):
        self.seq_len = seq_len or config.TEMPORAL_SEQ_LEN
        self.clips = []
        for label, sub in [(0, "real"), (1, "fake")]:
            for clip_dir in sorted(glob(os.path.join(root_dir, sub, "*"))):
                vid = os.path.basename(clip_dir)
                if video_ids is not None and vid not in video_ids:
                    continue
                frame_paths = sorted(glob(os.path.join(clip_dir, "*.jpg")))
                if len(frame_paths) >= 2:  # need at least a couple frames
                    self.clips.append((frame_paths, label))
        random.shuffle(self.clips)
        self.transform = temporal_transform()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frame_paths, label = self.clips[idx]
        frames = frame_paths[:self.seq_len]
        while len(frames) < self.seq_len:
            frames.append(frames[-1])

        tensors = []
        for p in frames:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(img))
        import torch
        clip_tensor = torch.stack(tensors)  # (T, C, H, W)
        return clip_tensor, float(label)


class FlatImageDataset(Dataset):
    """
    Flat real/fake image dataset for standalone image datasets that have
    NO video grouping -- e.g. GAN/diffusion-synthesized face datasets
    where each image is an independently generated sample, not a video
    frame. Unlike FrameDataset (which relies on load_video_split() +
    video_manifest.csv for a leak-free video-level split), this splits at
    the IMAGE level, stratified by class, seeded for reproducibility.

    Accepts one or more root_dirs (each with real/ and fake/ subfolders)
    so you can combine multiple synthesis datasets -- e.g. StyleGAN +
    DFFD + a diffusion-generated set -- into one training run.

    Caveat: if a dataset contains multiple crops/augmentations from the
    same source photo, those could end up split across train/val since
    there's no identity grouping here. Fine for independently-generated
    GAN samples (the common case); worth checking for datasets built from
    few source identities with many variants each.
    """

    def __init__(self, root_dirs, split: str = "train", val_ratio: float = 0.15,
                 seed: int = 42, img_size: int = None, transform=None):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        all_samples = []
        for root_dir in root_dirs:
            for label, sub_dir in [(0, "real"), (1, "fake")]:
                for path in glob(os.path.join(root_dir, sub_dir, "*")):
                    if os.path.isfile(path):
                        all_samples.append((path, label))

        rng = random.Random(seed)
        rng.shuffle(all_samples)

        real = [s for s in all_samples if s[1] == 0]
        fake = [s for s in all_samples if s[1] == 1]

        def split_list(items):
            n_val = max(1, int(len(items) * val_ratio))
            return items[n_val:], items[:n_val]

        real_train, real_val = split_list(real)
        fake_train, fake_val = split_list(fake)

        self.samples = (real_train + fake_train) if split == "train" else (real_val + fake_val)
        rng.shuffle(self.samples)
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"[FlatImageDataset] split={split} -> {len(self.samples)} samples "
              f"(real={n_real}, fake={n_fake}) from {len(root_dirs)} source dir(s)")

        self.transform = transform or spatial_transform(img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        return tensor, float(label)
