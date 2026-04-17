import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO

# YOLOv8-seg - segmentation variant, downloads automatically
_seg_model = None

def _load_model():
    global _seg_model
    if _seg_model is None:
        _seg_model = YOLO("yolov8n-seg.pt")
    return _seg_model


# Only segment objects we care about (subset of COCO classes)
TARGET_CLASSES = {"person", "car", "dog", "bicycle", "motorcycle", "bus", "truck"}

MASK_COLORS = [
    (255, 56,  56),   # red
    (56,  255, 56),   # green
    (56,  56,  255),  # blue
    (255, 255, 56),   # yellow
    (255, 56,  255),  # magenta
]


def segment_frame(frame_bgr: np.ndarray, conf: float = 0.35) -> dict:
    """
    Run segmentation on a single frame.
    Returns annotated frame + list of detected segments.
    """
    model   = _load_model()
    results = model(frame_bgr, conf=conf, verbose=False)[0]

    overlay   = frame_bgr.copy()
    segments  = []

    if results.masks is not None:
        for i, (mask, box) in enumerate(zip(results.masks.data, results.boxes)):
            label = model.names[int(box.cls)]
            if label not in TARGET_CLASSES:
                continue

            conf_val = float(box.conf)
            color    = MASK_COLORS[i % len(MASK_COLORS)]

            # apply mask as a coloured overlay
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(
                mask_np,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            overlay[mask_resized] = (
                overlay[mask_resized] * 0.5 +
                np.array(color) * 0.5
            ).astype(np.uint8)

            # bounding box + label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f"{label} {conf_val:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            segments.append({
                "label":      label,
                "confidence": round(conf_val, 3),
                "bbox":       [x1, y1, x2, y2],
            })

    return {"annotated": overlay, "segments": segments}


def segment_video(video_path: str, output_dir: str, every_n: int = 30) -> list[dict]:
    """Run segmentation on sampled frames, save annotated images."""
    os.makedirs(output_dir, exist_ok=True)

    cap       = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    rows      = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            result = segment_frame(frame)

            out_name = f"seg_frame_{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), result["annotated"])

            rows.append({
                "frame":    frame_idx,
                "second":   round(frame_idx / fps, 2),
                "segments": result["segments"],
                "count":    len(result["segments"]),
            })

        frame_idx += 1

    cap.release()
    return rows