"""
Face detection + alignment.

Default backend: MTCNN (facenet-pytorch) — pip-installable, ships its own
weights, runs well on 12GB VRAM, no manual model download required.

Optional backend: YOLOv8-face (ultralytics), swap in via
config.FACE_DETECTOR_BACKEND = "yolov8_face" if you already have a
face-tuned YOLOv8 checkpoint (note: your existing yolov8m_UrduDoc.pt is a
document-layout model, not a face model — it will NOT work here. You'd need
a separate face-detection checkpoint, e.g. from a community YOLOv8-face repo).
"""

import cv2
import numpy as np
import torch

import config


class FaceDetector:
    def __init__(self, backend: str = None, device: str = None):
        self.backend = backend or config.FACE_DETECTOR_BACKEND
        self.device = device or config.DEVICE
        self._model = None
        self._load_backend()

    def _load_backend(self):
        if self.backend == "mtcnn":
            from facenet_pytorch import MTCNN
            self._model = MTCNN(
                keep_all=True,
                thresholds=[0.6, 0.7, config.FACE_CONF_THRESHOLD],
                min_face_size=config.MIN_FACE_SIZE,
                device=self.device,
                post_process=False,
            )
        elif self.backend == "yolov8_face":
            from ultralytics import YOLO
            self._model = YOLO(config.YOLOV8_FACE_WEIGHTS)
        else:
            raise ValueError(f"Unknown face detector backend: {self.backend}")

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns a list of dicts: {"box": (x1,y1,x2,y2), "conf": float}
        Boxes are already margin-expanded and clipped to frame bounds.
        """
        h, w = frame_bgr.shape[:2]

        if self.backend == "mtcnn":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes, probs = self._model.detect(rgb)
            results = []
            if boxes is None:
                return results
            for box, prob in zip(boxes, probs):
                if prob is None or prob < config.FACE_CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = self._expand_and_clip(box, w, h)
                if (x2 - x1) < config.MIN_FACE_SIZE or (y2 - y1) < config.MIN_FACE_SIZE:
                    continue
                results.append({"box": (x1, y1, x2, y2), "conf": float(prob)})
            return results

        elif self.backend == "yolov8_face":
            preds = self._model.predict(frame_bgr, verbose=False, device=self.device)[0]
            results = []
            for box in preds.boxes:
                conf = float(box.conf[0])
                if conf < config.FACE_CONF_THRESHOLD:
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = self._expand_and_clip(xyxy, w, h)
                results.append({"box": (x1, y1, x2, y2), "conf": conf})
            return results

    @staticmethod
    def _expand_and_clip(box, frame_w, frame_h):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        mx, my = bw * config.FACE_MARGIN, bh * config.FACE_MARGIN
        x1, y1 = x1 - mx, y1 - my
        x2, y2 = x2 + mx, y2 + my
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(frame_w, int(x2)); y2 = min(frame_h, int(y2))
        return x1, y1, x2, y2

    def crop_faces(self, frame_bgr: np.ndarray, detections=None):
        """Convenience: detect (if needed) and return list of (crop_bgr, box, conf)."""
        detections = detections if detections is not None else self.detect(frame_bgr)
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append((crop, det["box"], det["conf"]))
        return crops
