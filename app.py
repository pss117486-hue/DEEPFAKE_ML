"""
Gradio UI for the offline deepfake + AI-image detection system.
Mirrors api.py's three endpoints:

  - Face-swap (Image): /analyze/image
  - Face-swap (Video): /analyze/video
  - General AI Image:  /analyze/general-image

Faces vs backgrounds: the two face-swap tabs detect MANIPULATED FACES in
otherwise-real photos/video (they will not flag an AI-generated landscape).
The General AI Image tab detects fully AI-GENERATED content of any kind --
landscapes, buildings, objects, scenes -- and does no face detection at
all. Different questions; use the tab matching what you're checking.

Every tab accepts a file upload OR a URL.

Usage:
    python app.py
"""
import os
import sys
import tempfile
from urllib.parse import urlparse

import cv2
import gradio as gr
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from inference.image_inference import ImageDeepfakeDetector
from inference.video_inference import VideoDeepfakeDetector
from models.pretrained_ai_detector import predict_image as predict_general_image
from models.fusion import confidence_label

config.apply_runtime_settings()

print("[app] loading face-swap models (this happens once)...")
image_detector = ImageDeepfakeDetector()
video_detector = VideoDeepfakeDetector()
print("[app] loaded. General AI detector loads lazily on first use.")

VERDICT_COLORS = {"real": "#1e8e3e", "fake": "#d93025"}
UNCERTAIN_COLOR = "#f9ab00"
DEFAULT_COLOR = "#5f6368"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CONTENT_TYPE_EXT_MAP = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/bmp": ".bmp", "image/webp": ".webp",
    "video/mp4": ".mp4", "video/quicktime": ".mov", "video/x-msvideo": ".avi",
    "video/webm": ".webm", "video/x-matroska": ".mkv",
}


def download_from_url(url, allowed_exts):
    url = url.strip()
    if not url:
        raise ValueError("No URL provided.")
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    resp = requests.get(url, stream=True, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    if ext not in allowed_exts:
        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        ext = CONTENT_TYPE_EXT_MAP.get(content_type, "")
    if ext not in allowed_exts:
        raise ValueError("Unsupported file type from URL (detected: '" + (ext or "none") + "').")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    for chunk in resp.iter_content(chunk_size=1 << 16):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def resolve_input(file_path, url, allowed_exts):
    if url and url.strip():
        return download_from_url(url, allowed_exts), True
    if file_path:
        return file_path, False
    raise ValueError("Provide either a file upload or a URL.")


def verdict_color(verdict):
    v = (verdict or "").lower()
    if "uncertain" in v or "review" in v:
        return UNCERTAIN_COLOR
    if "no face" in v:
        return DEFAULT_COLOR
    return VERDICT_COLORS.get(v, DEFAULT_COLOR)


def score_bar_html(label, score, color="#4285f4"):
    if score is None:
        return ("<div style='margin-bottom:10px;'><div style='display:flex;"
                "justify-content:space-between;font-size:14px;color:#333;'><span>"
                + label + "</span><span>n/a</span></div>"
                "<div style='background:#e8eaed;border-radius:6px;height:10px;'></div></div>")
    pct = max(0.0, min(1.0, float(score))) * 100
    return ("<div style='margin-bottom:10px;'><div style='display:flex;"
            "justify-content:space-between;font-size:14px;color:#333;'><span>"
            + label + "</span><span>" + f"{pct:.1f}" + "%</span></div>"
            "<div style='background:#e8eaed;border-radius:6px;height:10px;overflow:hidden;'>"
            "<div style='width:" + f"{pct:.1f}" + "%;background:" + color + ";height:100%;'></div>"
            "</div></div>")


def verdict_card_html(verdict, final_score):
    color = verdict_color(verdict)
    score_txt = (f"{float(final_score) * 100:.1f}% AI/fake-probability") if final_score is not None else ""
    return ("<div style='border:2px solid " + color + ";border-radius:12px;padding:20px;"
            "text-align:center;background:" + color + "15;margin-bottom:16px;'>"
            "<div style='font-size:28px;font-weight:700;color:" + color + ";"
            "text-transform:uppercase;letter-spacing:1px;'>" + str(verdict) + "</div>"
            "<div style='font-size:15px;color:#444;margin-top:6px;'>" + score_txt + "</div></div>")


def draw_face_boxes(frame_bgr, faces_info):
    annotated = frame_bgr.copy()
    for face in faces_info:
        x1, y1, x2, y2 = [int(v) for v in face["box"]]
        color_bgr = (0, 0, 255) if face["verdict"] == "fake" else \
                    (0, 165, 255) if "uncertain" in face["verdict"] else (0, 200, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 3)
        label = f"{face['verdict']} ({face['final_score']*100:.0f}%)"
        cv2.putText(annotated, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
    return annotated


def analyze_image(image_path, image_url):
    try:
        resolved_path, is_temp = resolve_input(image_path, image_url, IMAGE_EXTS)
    except ValueError as e:
        return None, "<div style='color:#d93025;'>" + str(e) + "</div>", ""
    try:
        result = image_detector.analyze(resolved_path)
        frame = cv2.imread(resolved_path)
        if result.get("faces_found", 0) == 0:
            annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            card = verdict_card_html(result["verdict"], result.get("final_score"))
            details = ("<div style='font-size:13px;color:#666;margin-bottom:8px;'>"
                       "No face detected - this tab only checks for manipulated FACES. "
                       "If you're checking whether the whole image is AI-generated, "
                       "use the 'General AI Image' tab instead.</div>"
                       + score_bar_html("Forensic (ELA/metadata)", result.get("forensic_score"), "#9334e6"))
            return annotated_rgb, card, details
        annotated = draw_face_boxes(frame, result["faces"])
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        card = verdict_card_html(result["verdict"], result["final_score"])
        worst_face = max(result["faces"], key=lambda f: f["final_score"])
        details = ("<div style='font-size:13px;color:#666;margin-bottom:8px;'>"
                   + str(result["faces_found"]) + " face(s) detected - showing most suspicious face</div>"
                   + score_bar_html("Spatial (visual artifacts)", worst_face.get("spatial_score"), "#4285f4")
                   + score_bar_html("Frequency (FFT artifacts)", worst_face.get("frequency_score"), "#fbbc04")
                   + score_bar_html("Forensic (ELA/metadata)", worst_face.get("forensic_score"), "#9334e6"))
        return annotated_rgb, card, details
    finally:
        if is_temp:
            os.remove(resolved_path)


def analyze_video(video_path, video_url):
    try:
        resolved_path, is_temp = resolve_input(video_path, video_url, VIDEO_EXTS)
    except ValueError as e:
        return "<div style='color:#d93025;'>" + str(e) + "</div>", ""
    try:
        result = video_detector.analyze(resolved_path)
        if result.get("faces_found", 0) == 0 and "final_score" in result and \
           result.get("identities_tracked") is None:
            card = verdict_card_html(result["verdict"], result.get("final_score"))
            details = score_bar_html("Forensic (container/metadata)", result.get("forensic_score"), "#9334e6")
            return card, details
        card = verdict_card_html(result["verdict"], result["final_score"])
        meta = ("<div style='font-size:13px;color:#666;margin-bottom:8px;'>"
                + str(result.get("identities_tracked", 0)) + " identit(y/ies) tracked across "
                + str(result.get("frames_analyzed", 0)) + " analyzed frames</div>")
        details = (meta
                   + score_bar_html("Spatial (visual artifacts)", result.get("spatial_score"), "#4285f4")
                   + score_bar_html("Frequency (FFT artifacts)", result.get("frequency_score"), "#fbbc04")
                   + score_bar_html("Temporal (inter-frame consistency)", result.get("temporal_score"), "#34a853")
                   + score_bar_html("Forensic (container/metadata)", result.get("forensic_score"), "#9334e6"))
        return card, details
    finally:
        if is_temp:
            os.remove(resolved_path)


def analyze_general(image_path, image_url):
    try:
        resolved_path, is_temp = resolve_input(image_path, image_url, IMAGE_EXTS)
    except ValueError as e:
        return None, "<div style='color:#d93025;'>" + str(e) + "</div>", ""
    try:
        raw = predict_general_image(resolved_path)
        label, confidence = confidence_label(raw["fake_probability"])
        frame = cv2.imread(resolved_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        card = verdict_card_html(label.lower(), raw["fake_probability"])
        per_model_html = ""
        for model_id, model_result in raw["per_model"].items():
            per_model_html += score_bar_html(model_id, model_result["fake_probability"], "#8e44ad")
        details = ("<div style='font-size:13px;color:#666;margin-bottom:8px;'>"
                   "Whole-image AI-generation check (no face detection) - "
                   + str(confidence) + "% confidence in '" + label + "'</div>"
                   + "<div style='font-size:12px;color:#555;margin:10px 0 6px;'><b>Per-model scores</b> "
                     "(disagreement between models means low confidence, not a bug):</div>"
                   + per_model_html
                   + "<div style='font-size:11px;color:#999;margin-top:12px;'>"
                     "Note: these pretrained models were trained on earlier-generation image "
                     "generators. Newer ones (Midjourney v6+, Flux, GPT Image) can evade them, "
                     "so a REAL verdict here is weaker evidence than a FAKE one.</div>")
        return frame_rgb, card, details
    finally:
        if is_temp:
            os.remove(resolved_path)


CUSTOM_CSS = """
.gradio-container {max-width: 1000px !important; margin: auto;}
footer {display: none !important;}
"""

with gr.Blocks(title="Deepfake & AI-Image Detector", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Offline Deepfake & AI-Image Detector

        **Checking a FACE?** Use the face-swap tabs - they detect manipulated
        faces in otherwise-real photos/video.
        **Checking a BACKGROUND / landscape / building / whole scene?** Use the
        General AI Image tab - it detects fully AI-generated content and does
        no face detection at all.

        Upload a file or paste a URL in any tab. Runs offline on your local GPU.
        """
    )

    with gr.Tabs():
        with gr.Tab("Face-swap: Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="filepath", label="Upload image", height=300)
                    image_url_input = gr.Textbox(label="...or paste an image URL", placeholder="https://example.com/photo.jpg")
                    image_button = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=1):
                    image_output = gr.Image(label="Annotated result", height=300)
            image_verdict = gr.HTML()
            image_details = gr.HTML()
            image_button.click(fn=analyze_image,
                               inputs=[image_input, image_url_input],
                               outputs=[image_output, image_verdict, image_details])

        with gr.Tab("Face-swap: Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload video", height=300)
                    video_url_input = gr.Textbox(label="...or paste a video URL", placeholder="https://example.com/clip.mp4")
                    video_button = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=1):
                    video_verdict = gr.HTML()
                    video_details = gr.HTML()
            video_button.click(fn=analyze_video,
                               inputs=[video_input, video_url_input],
                               outputs=[video_verdict, video_details])

        with gr.Tab("General AI Image (landscape / buildings / any scene)"):
            with gr.Row():
                with gr.Column(scale=1):
                    general_input = gr.Image(type="filepath", label="Upload image", height=300)
                    general_url_input = gr.Textbox(label="...or paste an image URL", placeholder="https://example.com/landscape.jpg")
                    general_button = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=1):
                    general_output = gr.Image(label="Image", height=300)
            general_verdict = gr.HTML()
            general_details = gr.HTML()
            general_button.click(fn=analyze_general,
                                 inputs=[general_input, general_url_input],
                                 outputs=[general_output, general_verdict, general_details])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
