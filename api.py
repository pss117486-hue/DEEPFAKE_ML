"""
FastAPI REST API for the offline deepfake + AI-image detection system.

Endpoints:
    GET  /health                  - health check, confirms models loaded
    POST /analyze/image           - face-swap detection on an image
    POST /analyze/video           - face-swap detection on a video
    POST /analyze/general-image   - general AI-generation detection
                                    (landscapes, buildings, any content --
                                    NOT face-specific)

Every analyze endpoint accepts EITHER an uploaded file (form field
"file") OR a URL (form field "url"), matching what app.py's Gradio UI
offers. URLs are downloaded to a temp file, analyzed, then cleaned up.

All responses include a clear `prediction` ("FAKE" / "REAL" /
"UNCERTAIN") and `confidence_percent` (0-100) on top of the raw scores.
For images with multiple faces, each per-face entry gets the same two
fields.

NOTE on which detector to use: /analyze/image and /analyze/video run the
face-SWAP ensemble (spatial/frequency/temporal/forensic) -- they detect
manipulated faces in real photos/video, and will NOT flag an AI-generated
landscape. /analyze/general-image runs a separate pretrained ensemble
that works on any content but is not face-specific. They answer different
questions; pick the one matching your input.

Usage:
    pip install fastapi uvicorn python-multipart
    python api.py
Then visit http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
import os
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse

import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from inference.image_inference import ImageDeepfakeDetector
from inference.video_inference import VideoDeepfakeDetector
from models.pretrained_ai_detector import predict_image as predict_general_image
from models.fusion import confidence_label

detectors = {}

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CONTENT_TYPE_EXT_MAP = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/bmp": ".bmp", "image/webp": ".webp",
    "video/mp4": ".mp4", "video/quicktime": ".mov", "video/x-msvideo": ".avi",
    "video/webm": ".webm", "video/x-matroska": ".mkv",
}


def _add_prediction_fields(result):
    """Adds prediction + confidence_percent to a result dict and to each
    per-face entry if present."""
    label, confidence = confidence_label(result["final_score"])
    result["prediction"] = label
    result["confidence_percent"] = confidence

    if "faces" in result:
        for face in result["faces"]:
            f_label, f_confidence = confidence_label(face["final_score"])
            face["prediction"] = f_label
            face["confidence_percent"] = f_confidence
    return result


def _download_from_url(url, allowed_exts):
    url = url.strip()
    if not url:
        raise ValueError("Empty URL provided.")

    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()

    resp = requests.get(url, stream=True, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    if ext not in allowed_exts:
        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        ext = CONTENT_TYPE_EXT_MAP.get(content_type, "")

    if ext not in allowed_exts:
        raise ValueError(
            "Couldn't determine a supported file type from this URL (detected: '"
            + (ext or "none") + "'). Expected one of: " + str(sorted(allowed_exts))
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    for chunk in resp.iter_content(chunk_size=1 << 16):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def _save_upload_to_temp(file, ext):
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


def _resolve_input(file, url, allowed_exts):
    """Returns (local_path, display_name). Raises HTTPException(400) on
    bad input so FastAPI returns a clean error rather than a 500."""
    if url and url.strip():
        try:
            path = _download_from_url(url, allowed_exts)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail="Failed to fetch URL: " + str(e))
        return path, url.strip()

    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file extension '" + ext + "'. Allowed: " + str(sorted(allowed_exts)),
            )
        return _save_upload_to_temp(file, ext), file.filename

    raise HTTPException(status_code=400, detail="Provide either a 'file' upload or a 'url'.")


@asynccontextmanager
async def lifespan(app):
    print("[api] loading models (this happens once, may take a moment)...")
    config.apply_runtime_settings()
    detectors["image"] = ImageDeepfakeDetector()
    detectors["video"] = VideoDeepfakeDetector()
    print("[api] face-swap models loaded. General AI detector loads lazily on first use.")
    yield
    detectors.clear()


app = FastAPI(
    title="Offline Deepfake & AI-Image Detector API",
    description="Face-swap detection (image/video) plus general AI-generation "
                "detection for any image content. Runs entirely offline.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok" if "image" in detectors and "video" in detectors else "loading",
        "device": config.DEVICE,
        "endpoints": ["/analyze/image", "/analyze/video", "/analyze/general-image"],
    }


@app.post("/analyze/image")
async def analyze_image(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)):
    """Face-SWAP detection on an image. Detects manipulated faces in real
    photos -- will NOT flag a fully AI-generated image; use
    /analyze/general-image for that."""
    if "image" not in detectors:
        raise HTTPException(status_code=503, detail="Model not yet loaded, try again shortly.")

    tmp_path, display_name = _resolve_input(file, url, ALLOWED_IMAGE_EXT)
    try:
        result = detectors["image"].analyze(tmp_path)
        result["image"] = display_name
        return JSONResponse(content=_add_prediction_fields(result))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Analysis failed: " + str(e))
    finally:
        os.remove(tmp_path)


@app.post("/analyze/video")
async def analyze_video(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)):
    """Face-SWAP detection on a video (spatial + frequency + temporal +
    forensic across sampled frames). Can take seconds to minutes depending
    on length and GPU."""
    if "video" not in detectors:
        raise HTTPException(status_code=503, detail="Model not yet loaded, try again shortly.")

    tmp_path, display_name = _resolve_input(file, url, ALLOWED_VIDEO_EXT)
    try:
        result = detectors["video"].analyze(tmp_path)
        result["video"] = display_name
        return JSONResponse(content=_add_prediction_fields(result))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Analysis failed: " + str(e))
    finally:
        os.remove(tmp_path)


@app.post("/analyze/general-image")
async def analyze_general_image(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)):
    """
    General AI-generation detection -- works on landscapes, buildings,
    objects, any content. NOT face-specific and does not use face
    detection. Runs an ensemble of pretrained models and returns both the
    averaged verdict and each model's individual score.

    CAVEAT: these pretrained models were trained on earlier-generation
    image generators. Newer generators (Midjourney v6+, Flux, GPT Image)
    can evade them -- a "REAL" verdict here is weaker evidence than a
    "FAKE" one. Treat as a signal, not a final answer.
    """
    tmp_path, display_name = _resolve_input(file, url, ALLOWED_IMAGE_EXT)
    try:
        raw = predict_general_image(tmp_path)
        label, confidence = confidence_label(raw["fake_probability"])
        return JSONResponse(content={
            "image": display_name,
            "fake_probability": raw["fake_probability"],
            "prediction": label,
            "confidence_percent": confidence,
            "per_model": raw["per_model"],
            "note": "Pretrained ensemble, not face-specific. Trained on earlier-generation "
                    "generators -- newest generators may evade detection.",
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Analysis failed: " + str(e))
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
