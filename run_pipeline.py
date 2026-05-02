import argparse
import os
from src.utils.file_utils import list_videos, ensure_dirs
from src.vision.detect_objects  import detect_objects
from src.vision.classify_scene  import classify_video
from src.vision.segment_objects import segment_video
from src.audio.transcribe       import transcribe_video
from src.temporal.aggregator    import (
    get_scene_transitions,
    get_object_stats,
    get_event_durations,
    print_temporal_report,
)
from src.fusion.summarize import generate_summary


def run(video_path):
    name     = os.path.basename(video_path)
    stem     = os.path.splitext(name)[0]
    fps      = 30  # default; detect_objects could return this too

    print(f"\n{'='*55}\nProcessing: {name}\n{'='*55}")

    # Object detection
    print("- Running YOLOv8 detection...")
    detections = detect_objects(video_path, every_n=30)

    # Scene classification
    print("- Running scene classification...")
    scene_rows = classify_video(video_path, every_n=30)

    # Segmentation (key objects only)
    print("- Running segmentation...")
    seg_out = os.path.join("outputs", "segmented_frames", stem)
    segment_video(video_path, seg_out, every_n=30)

    # Temporal aggregation
    transitions  = get_scene_transitions(scene_rows)
    object_stats = get_object_stats(detections)
    person_events = get_event_durations(detections, "person", fps=fps)

    print_temporal_report(transitions, object_stats)

    if person_events:
        print("\n--- Person on screen ---")
        for e in person_events:
            print(f"  {e['start_sec']}s → {e['end_sec']}s  ({e['duration_sec']}s)")

    # Transcription
    print("\n- Running Whisper...")
    audio  = transcribe_video(video_path)
    print(f"  Transcript: {audio['transcript'] or '[no speech]'}")

    # Fusion
    summary = generate_summary(
        video_name     = name,
        detection_rows = detections,
        transcript     = audio["transcript"],
        scene_rows     = scene_rows,
        transitions    = transitions,
        object_stats   = object_stats,
    )
    print(f"\n{summary}")

    # Save
    out_file = os.path.join("outputs", f"{stem}_summary.txt")
    with open(out_file, "w") as f:
        f.write(summary)
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    ensure_dirs()
    os.makedirs("outputs/segmented_frames", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_videos")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        run(args.input)
    else:
        for path in list_videos(args.input):
            run(path)
