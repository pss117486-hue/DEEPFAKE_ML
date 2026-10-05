"""
General-purpose AI-generated image detector -- ENSEMBLE of two pretrained
models to cover more generator families.

  1. Organika/sdxl-detector -- binary real/fake, trained on SDXL output.
     Good precision on SDXL, does not generalize to other generators.

  2. Ateeqq/ai-vs-human-image-detector -- SigLIP-based binary classifier.
     Replaces two earlier attempts: yaya36095/ai-source-detector (broken
     checkpoint on Hugging Face -- config declares ViT but the actual
     weights are RoBERTa, so transformers silently discards them and
     randomly initializes the model, producing pure noise -- caught via
     the per-model breakdown before trusting it) and Dafilab/ai-image-detector
     (ambiguous/inconsistent checkpoint filename across its own docs,
     couldn't reliably resolve which file to download). Both standard
     transformers models now (Organika + Ateeqq), loaded the same way,
     for consistency and lower integration risk.

Combined by simple averaging of each model's fake_probability. If one
model is confident and the other isn't, the average lands closer to
"uncertain" rather than confidently wrong.

IMPORTANT CAVEAT: neither model has been independently validated here
against a genuinely held-out generator the way this project's other
branches were. Expect blind spots against generators released after
either model's training data.

First run downloads both models (~350MB + ~370MB) from Hugging Face;
cached locally afterward.
"""

import torch
from PIL import Image

import config

_MODELS = {}


def _load_model(model_id, device):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    if model_id not in _MODELS:
        print("[pretrained-ai] loading " + model_id + " (downloads once, cached locally after)...")
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        model.to(device)
        model.eval()
        print("[pretrained-ai] loaded " + model_id + ". labels: " + str(model.config.id2label))
        _MODELS[model_id] = (model, processor)
    return _MODELS[model_id]


def load_pretrained_ai_detector(device=None):
    device = device or config.DEVICE
    _load_model("Organika/sdxl-detector", device)
    _load_model("Ateeqq/ai-vs-human-image-detector", device)
    return _MODELS


@torch.no_grad()
def _score_one_model(model_id, image, device):
    model, processor = _load_model(model_id, device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    id2label = model.config.id2label
    label_probs = {id2label[i].lower(): p for i, p in enumerate(probs)}

    real_keys = [k for k in label_probs if k == "real" or "real" in k or "human" in k]
    if real_keys:
        fake_probability = 1.0 - sum(label_probs[k] for k in real_keys)
    else:
        fake_keys = [k for k in label_probs if "artificial" in k or "fake" in k or "ai" in k]
        fake_probability = label_probs[fake_keys[0]] if fake_keys else (probs[1] if len(probs) > 1 else probs[0])

    return {"fake_probability": float(fake_probability), "raw_labels": label_probs}


def predict_image(image_path, device=None):
    device = device or config.DEVICE
    image = Image.open(image_path).convert("RGB")

    per_model = {
        "Organika/sdxl-detector": _score_one_model("Organika/sdxl-detector", image, device),
        "Ateeqq/ai-vs-human-image-detector": _score_one_model("Ateeqq/ai-vs-human-image-detector", image, device),
    }

    ensemble_score = sum(m["fake_probability"] for m in per_model.values()) / len(per_model)

    return {"fake_probability": ensemble_score, "per_model": per_model}
