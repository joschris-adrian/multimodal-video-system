import os

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

def list_videos(folder):
    folder = os.path.abspath(folder)
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ]

def ensure_dirs():
    for d in ["temp/frames", "temp", "outputs/annotated_frames", "models"]:
        os.makedirs(d, exist_ok=True)