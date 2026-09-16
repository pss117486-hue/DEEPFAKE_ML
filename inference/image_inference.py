"""
End-to-end image inference: face detect -> spatial branch -> frequency
branch -> gan branch -> forensics -> fusion -> verdict.

Usage:
    python -m inference.image_inference --image path/to/image.jpg
"""

import argparse
import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from face_detector import FaceDetector
from models.spatial_classifier import load_spatial_model, predict_batch as spatial_predict
from models.frequency_branch import load_frequency_model, predict_batch as freq_predict
from models.gan_classifier import load_gan_model, predict_batch as gan_predict
from models.fusion import load_fusion_model, fuse_scores, classify
from forensics.metadata_ela import forensic_score


class ImageDeepfakeDetector:
    def __init__(self):
        config.apply_runtime_settings()
        self.face_detector = FaceDetector()
        self.spatial_model = load_spatial_model()
        self.freq_model = load_frequency_model()
        # GAN/diffusion branch disabled by default -- see config.GAN_BRANCH_ENABLED
        # for why (fails to generalize past its own training source; not
        # currently reliable enough to influence verdicts). Skipping the load
        # entirely when disabled also saves VRAM/startup time.
        self.gan_model = load_gan_model() if config.GAN_BRANCH_ENABLED else None
        self.fusion_model = load_fusion_model()

    def analyze(self, image_path: str) -> dict:
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = self.face_detector.crop_faces(frame)
        forensic = forensic_score(image_path, is_video=False)

        if not faces:
            return {
                "image": image_path,
                "faces_found": 0,
                "verdict": "no face detected — forensic-only analysis",
                "forensic_score": forensic,
                "final_score": forensic,
            }

        crops = [c for c, _, _ in faces]
        spatial_scores = spatial_predict(self.spatial_model, crops)
        freq_scores = freq_predict(self.freq_model, crops)
        gan_scores = gan_predict(self.gan_model, crops) if config.GAN_BRANCH_ENABLED else [None] * len(crops)

        per_face = []
        for i, (crop, box, conf) in enumerate(faces):
            s_score = spatial_scores[i]
            f_score = freq_scores[i]
            g_score = gan_scores[i]
            fused = fuse_scores(
                spatial=s_score, frequency=f_score, temporal=None, forensic=forensic,
                gan=g_score, fusion_model=self.fusion_model,
            )
            per_face.append({
                "box": box,
                "detector_confidence": conf,
                "spatial_score": s_score,
                "frequency_score": f_score,
                "gan_score": g_score,
                "forensic_score": forensic,
                "final_score": fused,
                "verdict": classify(fused),
            })

        overall_score = max(f["final_score"] for f in per_face)  # most-suspicious face drives verdict
        return {
            "image": image_path,
            "faces_found": len(faces),
            "faces": per_face,
            "final_score": overall_score,
            "verdict": classify(overall_score),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    detector = ImageDeepfakeDetector()
    result = detector.analyze(args.image)

    print(f"\n=== {result['image']} ===")
    print(f"Faces found: {result['faces_found']}")
    print(f"Final score: {result['final_score']:.3f}  ->  {result['verdict']}")
    if "faces" in result:
        for i, f in enumerate(result["faces"]):
            gan_str = f"{f['gan_score']:.3f}" if f['gan_score'] is not None else "disabled"
            print(f"  Face {i}: spatial={f['spatial_score']:.3f} "
                  f"freq={f['frequency_score']:.3f} gan={gan_str} "
                  f"forensic={f['forensic_score']:.3f} "
                  f"final={f['final_score']:.3f} ({f['verdict']})")


if __name__ == "__main__":
    main()
