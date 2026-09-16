"""
Fusion layer: combines branch-level fake-probabilities into one calibrated
final score. Two modes:

1. Weighted average (default, works immediately, no training needed) --
   uses config.FUSION_FALLBACK_WEIGHTS.
2. Learned MLP (config.FUSION_CHECKPOINT) -- trained on held-out branch
   predictions via train/train_fusion.py. Generalizes better once you have
   even a modest labeled validation set, because it learns which branches
   to trust more for a given content type.

Input order: [spatial, frequency, temporal, forensic, gan].
NOTE: this is a 5-input fusion head (added `gan` for whole-face GAN/
diffusion-synthesis detection, alongside the original swap-detection
branches). Any FUSION_CHECKPOINT trained before the gan branch was added
was trained on 4 inputs and is INCOMPATIBLE with this fusion.py --
retrain via train/train_fusion.py on a CSV that includes a `gan` column,
or the model will raise a shape-mismatch error loading old weights.
"""

import os
import torch
import torch.nn as nn

import config


class FusionMLP(nn.Module):
    """Input order: [spatial, frequency, temporal, forensic, gan] scores in [0,1]."""

    def __init__(self, n_inputs: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_fusion_model(device: str = None):
    device = device or config.DEVICE
    checkpoint_path = config.FUSION_CHECKPOINT
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state["model_state"] if "model_state" in state else state
        # Infer n_inputs from the saved weights themselves rather than
        # assuming a fixed default -- correctly handles a 4-input
        # checkpoint (gan disabled, current default) or a 5-input one
        # (gan enabled) without needing this function edited either way.
        n_inputs = state_dict["net.0.weight"].shape[1]
        model = FusionMLP(n_inputs=n_inputs)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"[fusion] loaded learned fusion head from {checkpoint_path} "
              f"({n_inputs}-input, feature_cols={state.get('feature_cols', 'unknown')})")
        return model
    print("[fusion] no trained fusion head found -- using fallback weighted average "
          f"{config.FUSION_FALLBACK_WEIGHTS}")
    return None


def fuse_scores(spatial: float, frequency: float, temporal: float, forensic: float,
                 gan: float = None, fusion_model=None, device: str = None) -> float:
    """
    Any branch can be None if it wasn't run (e.g. temporal is skipped for
    single images; gan is None whenever config.GAN_BRANCH_ENABLED is
    False). Missing branches are excluded and remaining weights
    renormalized for the fallback path.

    When a learned fusion_model is provided, the input vector is built to
    match whatever that MODEL was actually trained on (4 inputs while gan
    is disabled, 5 if it was trained with gan included) -- inferred from
    the model's own first-layer size, not assumed from this function's
    arguments. This means a fusion checkpoint trained before/after the gan
    branch existed both work correctly without code changes here.
    """
    device = device or config.DEVICE
    scores = {"spatial": spatial, "frequency": frequency, "temporal": temporal,
              "forensic": forensic, "gan": gan}

    if fusion_model is not None:
        expected_inputs = fusion_model.net[0].in_features
        feature_order = ["spatial", "frequency", "temporal", "forensic", "gan"][:expected_inputs]
        if all(scores[k] is not None for k in feature_order):
            x = torch.tensor([[scores[k] for k in feature_order]], dtype=torch.float32).to(device)
            with torch.no_grad():
                logit = fusion_model(x)
            return float(torch.sigmoid(logit).cpu().item())

    # Fallback: weighted average over available branches
    names = ["spatial", "frequency", "temporal", "forensic", "gan"]
    weights = dict(zip(names, config.FUSION_FALLBACK_WEIGHTS))
    total_w, weighted_sum = 0.0, 0.0
    for name in names:
        val = scores[name]
        if val is None:
            continue
        w = weights[name]
        weighted_sum += w * val
        total_w += w
    if total_w == 0:
        return 0.5
    return weighted_sum / total_w


def classify(score: float) -> str:
    lo, hi = config.REVIEW_BAND
    if lo <= score <= hi:
        return "uncertain — needs human review"
    return "fake" if score >= config.DECISION_THRESHOLD else "real"


def confidence_label(score: float):
    """
    Returns (label, confidence_percent) for clear, human-readable output --
    used by the API to make results unambiguous at a glance instead of
    requiring the caller to interpret a raw probability + threshold.

    label: "FAKE", "REAL", or "UNCERTAIN" -- same underlying logic as
    classify(), just normalized to a clean constant for API consumers.
    confidence_percent: how confident the score is in whichever class it
    leans toward (max(score, 1-score) as a percentage). Meaningful even
    for "UNCERTAIN" results, where it will naturally sit close to 50%.
    """
    verdict = classify(score)
    if "uncertain" in verdict:
        label = "UNCERTAIN"
    elif "fake" in verdict:
        label = "FAKE"
    else:
        label = "REAL"
    confidence_percent = round(max(score, 1 - score) * 100, 1)
    return label, confidence_percent