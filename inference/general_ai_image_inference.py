"""
Standalone CLI for the general-purpose (non-face) AI-image detector --
now an ensemble of two pretrained models. Use this for landscapes,
buildings, and general scenes. For face-swap detection, use
inference/image_inference.py instead.

Usage:
    python -m inference.general_ai_image_inference --image path/to/photo.jpg
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.pretrained_ai_detector import predict_image
from models.fusion import confidence_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    config.apply_runtime_settings()
    result = predict_image(args.image)
    label, confidence = confidence_label(result["fake_probability"])

    print("\n=== " + args.image + " ===")
    print("ENSEMBLE fake_probability: " + f"{result['fake_probability']:.4f}")
    print("prediction: " + label + "   confidence: " + str(confidence) + "%")
    print("\nPer-model breakdown:")
    for model_id, model_result in result["per_model"].items():
        print("  " + model_id + ":")
        print("    fake_probability: " + f"{model_result['fake_probability']:.4f}")
        print("    raw labels: " + str(model_result["raw_labels"]))


if __name__ == "__main__":
    main()
