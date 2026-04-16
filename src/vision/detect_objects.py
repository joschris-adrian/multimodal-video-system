import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(video_path, conf=0.25, every_n=30):
    """
    Run YOLOv8 on a video. Returns list of per-frame detections
    and saves annotated frames to outputs/annotated_frames/.
    No ffmpeg needed — uses OpenCV.
    """
    annotated_dir = os.path.join("outputs", "annotated_frames",
                                 os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(annotated_dir, exist_ok=True)

    cap       = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    rows      = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            results  = model(frame, conf=conf, verbose=False)[0]
            labels   = [model.names[int(b.cls)] for b in results.boxes] if results.boxes else []

            # save annotated frame
            annotated = results.plot()
            out_name  = f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(annotated_dir, out_name), annotated)

            rows.append({
                "frame":         frame_idx,
                "second":        round(frame_idx / fps, 2),
                "labels":        labels,
                "unique_labels": list(set(labels)),
                "num_objects":   len(labels),
            })

        frame_idx += 1

    cap.release()
    return rows


def temporal_summary(rows, window=30):
    """
    Segment detections into time windows and print summary.
    Example: Frame 0-30: person, car
    """
    from collections import Counter

    buckets = {}
    for r in rows:
        bucket = (r["frame"] // window) * window
        buckets.setdefault(bucket, []).extend(r["labels"])

    print("\n--- Temporal Summary ---")
    for start, labels in sorted(buckets.items()):
        top = [obj for obj, _ in Counter(labels).most_common(3)]
        print(f"  Frame {start:>5}–{start+window:<5}: {', '.join(top) or '[nothing]'}")

    return buckets