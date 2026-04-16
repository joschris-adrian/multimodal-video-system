from collections import Counter

def generate_summary(video_name, detection_rows, transcript):
    """Combine vision + audio into a human-readable summary."""

    all_labels = [l for r in detection_rows for l in r["labels"]]
    top3       = [obj for obj, _ in Counter(all_labels).most_common(3)]
    empty_pct  = sum(1 for r in detection_rows if r["num_objects"] == 0) / len(detection_rows) * 100 if detection_rows else 100

    vision_str = f"Detected: {', '.join(top3)}" if top3 else "No objects detected"
    audio_str  = f"Audio: {transcript}" if transcript.strip() else "No speech detected"

    summary = (
        f"FILE: {video_name}\n"
        f"{vision_str}\n"
        f"{audio_str}\n"
        f"Empty frames: {empty_pct:.1f}%\n"
    )
    return summary