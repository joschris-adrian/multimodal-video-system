import cv2
import numpy as np
import os
from ultralytics import YOLO

_model = None

def _load_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


class SimpleTracker:
    """
    Lightweight IoU tracker — same idea as DeepSORT without the Reid model.
    Assigns persistent IDs to bounding boxes across frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 10):
        self.iou_threshold = iou_threshold
        self.max_lost      = max_lost
        self.tracks        = {}   # id → {box, label, lost}
        self.next_id       = 0

    def _iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
        return inter / union

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: [{"bbox": [x1,y1,x2,y2], "label": str, "conf": float}]
        returns:    same list with "track_id" added
        """
        matched_ids = set()
        results     = []

        for det in detections:
            best_id  = None
            best_iou = self.iou_threshold

            for tid, track in self.tracks.items():
                iou = self._iou(det["bbox"], track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_id  = tid

            if best_id is not None:
                self.tracks[best_id] = {
                    "box":   det["bbox"],
                    "label": det["label"],
                    "lost":  0,
                }
                matched_ids.add(best_id)
                results.append({**det, "track_id": best_id})
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "box":   det["bbox"],
                    "label": det["label"],
                    "lost":  0,
                }
                results.append({**det, "track_id": tid})

        # increment lost counter for unmatched tracks
        for tid in list(self.tracks):
            if tid not in matched_ids:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return results


def track_video(
    video_path: str,
    output_path: str,
    conf: float = 0.25,
    every_n: int = 1,          # track every frame for smooth IDs
) -> list[dict]:
    """
    Run detection + IoU tracking on a video.
    Saves annotated video with persistent track IDs.
    Returns per-frame tracking rows.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model   = _load_model()
    tracker = SimpleTracker()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    all_rows  = []

    COLORS = {}

    def get_color(tid):
        if tid not in COLORS:
            np.random.seed(tid * 7)
            COLORS[tid] = tuple(np.random.randint(50, 255, 3).tolist())
        return COLORS[tid]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)[0]
        dets    = []

        if results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets.append({
                    "bbox":  [x1, y1, x2, y2],
                    "label": model.names[int(box.cls)],
                    "conf":  float(box.conf),
                })

        tracked = tracker.update(dets)

        for t in tracked:
            x1, y1, x2, y2 = t["bbox"]
            tid    = t["track_id"]
            color  = get_color(tid)
            label  = f"{t['label']} #{tid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(frame)
        all_rows.append({
            "frame":   frame_idx,
            "second":  round(frame_idx / fps, 2),
            "tracked": tracked,
        })
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Tracked video saved → {output_path}")
    return all_rows