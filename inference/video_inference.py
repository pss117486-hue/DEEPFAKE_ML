"""
End-to-end video inference: sample frames -> detect + track faces ->
per-frame spatial/frequency/gan branches -> temporal branch over each
tracked identity's sequence -> container forensics -> fusion -> verdict.

The gan branch runs per sampled frame (like spatial/frequency) rather than
per-sequence, since whole-face synthesis artifacts are a per-frame visual
property, not an inter-frame temporal one.

Simple centroid-distance tracker is used to keep face-crop sequences
associated with the same identity across frames (good enough for
single/few-speaker clips; swap in a stronger tracker like ByteTrack if you
need robust multi-person tracking in crowded footage).

Usage:
    python -m inference.video_inference --video path/to/video.mp4
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from face_detector import FaceDetector
from models.spatial_classifier import load_spatial_model, predict_batch as spatial_predict
from models.frequency_branch import load_frequency_model, predict_batch as freq_predict
from models.gan_classifier import load_gan_model, predict_batch as gan_predict
from models.temporal_branch import load_temporal_model, predict_sequence as temporal_predict
from models.fusion import load_fusion_model, fuse_scores, classify
from forensics.metadata_ela import forensic_score


class SimpleTracker:
    """Minimal centroid tracker: good enough to group face crops by identity
    across a short clip without pulling in a full tracking dependency."""

    def __init__(self, max_distance: int = 80):
        self.max_distance = max_distance
        self.tracks = {}   # id -> {"centroid": (x,y), "crops": [...]}
        self.next_id = 0

    @staticmethod
    def _centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections):
        """detections: list of (crop, box, conf). Returns dict track_id -> crop for this frame."""
        assigned = {}
        used_tracks = set()

        for crop, box, conf in detections:
            c = self._centroid(box)
            best_id, best_dist = None, self.max_distance
            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue
                d = np.hypot(c[0] - track["centroid"][0], c[1] - track["centroid"][1])
                if d < best_dist:
                    best_id, best_dist = tid, d

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {"centroid": c, "crops": []}

            self.tracks[best_id]["centroid"] = c
            self.tracks[best_id]["crops"].append(crop)
            used_tracks.add(best_id)
            assigned[best_id] = crop

        return assigned


class VideoDeepfakeDetector:
    def __init__(self):
        config.apply_runtime_settings()
        self.face_detector = FaceDetector()
        self.spatial_model = load_spatial_model()
        self.freq_model = load_frequency_model()
        # GAN/diffusion branch disabled by default -- see config.GAN_BRANCH_ENABLED
        # for why (fails to generalize past its own training source; not
        # currently reliable enough to influence verdicts).
        self.gan_model = load_gan_model() if config.GAN_BRANCH_ENABLED else None
        self.temporal_model = load_temporal_model()
        self.fusion_model = load_fusion_model()

    def _sample_frame_indices(self, cap) -> list:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(src_fps / config.FRAME_SAMPLE_FPS)))
        indices = list(range(0, total_frames, step))
        return indices[:config.MAX_FRAMES_PER_VIDEO]

    def analyze(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        indices_to_sample = set(self._sample_frame_indices(cap))
        tracker = SimpleTracker()

        all_spatial_scores, all_freq_scores, all_gan_scores = [], [], []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices_to_sample:
                detections = self.face_detector.crop_faces(frame)
                tracker.update(detections)
                if detections:
                    crops = [c for c, _, _ in detections]
                    all_spatial_scores.extend(spatial_predict(self.spatial_model, crops))
                    all_freq_scores.extend(freq_predict(self.freq_model, crops))
                    if config.GAN_BRANCH_ENABLED:
                        all_gan_scores.extend(gan_predict(self.gan_model, crops))
            frame_idx += 1
        cap.release()

        forensic = forensic_score(video_path, is_video=True)

        if not tracker.tracks:
            return {
                "video": video_path,
                "faces_found": 0,
                "verdict": "no face detected — forensic-only analysis",
                "forensic_score": forensic,
                "final_score": forensic,
            }

        # Temporal branch: run per tracked identity, take the most suspicious.
        temporal_scores = []
        for tid, track in tracker.tracks.items():
            t_score = temporal_predict(self.temporal_model, track["crops"])
            temporal_scores.append(t_score)

        spatial_agg = float(np.mean(all_spatial_scores)) if all_spatial_scores else None
        freq_agg = float(np.mean(all_freq_scores)) if all_freq_scores else None
        gan_agg = float(np.mean(all_gan_scores)) if all_gan_scores else None
        temporal_agg = float(np.max(temporal_scores)) if temporal_scores else None

        fused = fuse_scores(
            spatial=spatial_agg, frequency=freq_agg, temporal=temporal_agg, forensic=forensic,
            gan=gan_agg, fusion_model=self.fusion_model,
        )

        return {
            "video": video_path,
            "identities_tracked": len(tracker.tracks),
            "frames_analyzed": len(all_spatial_scores),
            "spatial_score": spatial_agg,
            "frequency_score": freq_agg,
            "gan_score": gan_agg,
            "temporal_score": temporal_agg,
            "forensic_score": forensic,
            "final_score": fused,
            "verdict": classify(fused),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()

    detector = VideoDeepfakeDetector()
    result = detector.analyze(args.video)

    print(f"\n=== {result['video']} ===")
    for k, v in result.items():
        if k == "video":
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
