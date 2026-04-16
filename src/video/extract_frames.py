import cv2
import os

def extract_frames(video_path, output_dir, every_n=30):
    """Extract 1 frame every N frames using OpenCV. No ffmpeg needed."""
    os.makedirs(output_dir, exist_ok=True)

    cap       = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    saved     = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            fname = f"frame_{frame_idx:05d}.jpg"
            out   = os.path.join(output_dir, fname)
            cv2.imwrite(out, frame)
            saved.append({
                "frame_idx": frame_idx,
                "second":    round(frame_idx / fps, 2),
                "path":      out
            })

        frame_idx += 1

    cap.release()
    return saved