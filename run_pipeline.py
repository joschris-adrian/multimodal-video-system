import argparse
import os
from src.utils.file_utils import list_videos, ensure_dirs
from src.vision.detect_objects import detect_objects, temporal_summary
from src.audio.transcribe import transcribe_video
from src.fusion.summarize import generate_summary

def run(video_path):
    name = os.path.basename(video_path)
    print(f"\n{'='*50}\nProcessing: {name}\n{'='*50}")

    # 1. Vision
    print("Running YOLOv8...")
    detections = detect_objects(video_path, every_n=30)
    temporal_summary(detections)

    # 2. Audio
    print("\nRunning Whisper...")
    audio_result = transcribe_video(video_path)
    print(f"Transcript: {audio_result['transcript'] or '[no speech]'}")

    # 3. Fusion
    summary = generate_summary(name, detections, audio_result["transcript"])
    print(f"\n{summary}")

    # 4. Save
    out_file = os.path.join("outputs", f"{os.path.splitext(name)[0]}_summary.txt")
    with open(out_file, "w") as f:
        f.write(summary)
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    ensure_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_videos", help="Video file or folder")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        run(args.input)
    else:
        for path in list_videos(args.input):
            run(path)