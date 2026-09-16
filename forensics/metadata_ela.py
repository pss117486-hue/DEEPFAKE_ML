"""
Cheap, fast forensic signals that don't need a GPU or trained weights:

1. Error Level Analysis (ELA) — recompress the image at a known JPEG
   quality and diff against the original. Regions that were spliced/edited
   in tend to show a different error level than untouched regions because
   they carry a different original compression history.
2. Metadata inspection — missing/inconsistent EXIF, re-encoding signatures,
   double-JPEG-compression artifacts (naive giveaways some fakes still have).

These feed the "forensic" slot in the fusion layer.
"""

import io
import os
import subprocess
import numpy as np
from PIL import Image, ExifTags


def error_level_analysis(image_path: str, quality: int = 90) -> float:
    """
    Returns a scalar 'ELA score' — higher suggests localized recompression
    inconsistency (possible splice/edit). Not a probability by itself, feed
    it into the fusion layer's forensic slot after normalizing.
    """
    original = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    orig_arr = np.asarray(original).astype(np.int16)
    recompressed_arr = np.asarray(recompressed).astype(np.int16)
    diff = np.abs(orig_arr - recompressed_arr)

    # Normalize: mean diff across pixels, scaled into a rough [0,1] band.
    # Calibrate the divisor against your own real/fake sample set for best results.
    score = float(diff.mean()) / 20.0
    return min(score, 1.0)


def check_image_metadata(image_path: str) -> dict:
    """Lightweight EXIF presence/consistency check for images."""
    result = {"has_exif": False, "software_tag": None, "suspicious": False}
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            result["has_exif"] = True
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
            software = exif.get("Software")
            if software:
                result["software_tag"] = str(software)
                # Common AI-gen / editing tool signatures worth flagging for review
                flagged_tools = ["photoshop", "gimp", "stable diffusion", "midjourney"]
                if any(tool in str(software).lower() for tool in flagged_tools):
                    result["suspicious"] = True
        else:
            # No EXIF at all is itself mildly suspicious for a "camera original" claim,
            # but very common for legitimately re-saved/re-shared images — weak signal only.
            result["suspicious"] = False
    except Exception:
        pass
    return result


def check_video_container(video_path: str) -> dict:
    """
    Uses ffprobe (bundled with ffmpeg) to inspect codec/container metadata.
    Multiple encoder generations in one file, or an encoder tag associated
    with common face-swap tools, is a mild forensic signal.
    """
    result = {"encoder": None, "nb_streams": None, "suspicious": False}
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, timeout=30,
        )
        import json
        info = json.loads(proc.stdout)
        fmt = info.get("format", {})
        tags = fmt.get("tags", {})
        result["encoder"] = tags.get("encoder")
        result["nb_streams"] = fmt.get("nb_streams")
        if result["encoder"]:
            flagged = ["deepfacelab", "faceswap", "dfl"]
            if any(f in result["encoder"].lower() for f in flagged):
                result["suspicious"] = True
    except Exception:
        pass
    return result


def forensic_score(media_path: str, is_video: bool = False) -> float:
    """Combines the above into a single [0,1] forensic score for fusion."""
    signals = []
    if not is_video:
        try:
            signals.append(error_level_analysis(media_path))
        except Exception:
            pass
        meta = check_image_metadata(media_path)
        signals.append(0.7 if meta["suspicious"] else 0.1)
    else:
        meta = check_video_container(media_path)
        signals.append(0.7 if meta["suspicious"] else 0.1)

    if not signals:
        return 0.1  # neutral-low default when no signal could be computed
    return float(np.mean(signals))
