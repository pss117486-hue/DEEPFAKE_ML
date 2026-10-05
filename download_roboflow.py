"""
Downloads the Roboflow "Deepfake detection" classification dataset.

Requires ROBOFLOW_API_KEY set as an environment variable (never hardcode
API keys in a file that gets committed to git):
    setx ROBOFLOW_API_KEY "your_key_here"   (then restart your terminal)

Usage:
    python download_roboflow.py
"""
import os
from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the ROBOFLOW_API_KEY environment variable first (see docstring above).")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("deepfake-q1kew").project("deepfake-detection-h3was")
dataset = project.version(1).download("folder")

print(f"\nDownloaded to: {dataset.location}")
