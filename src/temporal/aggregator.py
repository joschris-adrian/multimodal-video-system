from collections import Counter


def get_scene_transitions(scene_rows: list[dict], min_duration: int = 2) -> list[dict]:
    """
    Collapse consecutive identical scenes into transitions.
    Filters out flickers shorter than min_duration frames.

    Input:  [{"frame": 0, "scene": "street"}, {"frame": 30, "scene": "indoor"}, ...]
    Output: [{"start": 0, "end": 90, "scene": "street"}, ...]
    """
    if not scene_rows:
        return []

    transitions = []
    current     = scene_rows[0]["scene"]
    start_frame = scene_rows[0]["frame"]
    count       = 1

    for row in scene_rows[1:]:
        if row["scene"] == current:
            count += 1
        else:
            if count >= min_duration:
                transitions.append({
                    "scene":       current,
                    "start_frame": start_frame,
                    "end_frame":   row["frame"],
                    "duration_frames": row["frame"] - start_frame,
                })
            current     = row["scene"]
            start_frame = row["frame"]
            count       = 1

    # last segment
    transitions.append({
        "scene":           current,
        "start_frame":     start_frame,
        "end_frame":       scene_rows[-1]["frame"],
        "duration_frames": scene_rows[-1]["frame"] - start_frame,
    })

    return transitions


def get_object_stats(detection_rows: list[dict]) -> dict:
    """
    Compute object-level statistics across all frames.

    Returns:
        most_common   - top 5 objects by total count
        persistent    - objects appearing in >50% of frames
        sporadic      - objects appearing in <10% of frames
        per_object    - frame count + presence % for each label
    """
    total = len(detection_rows)
    if total == 0:
        return {}

    all_labels   = [l for r in detection_rows for l in r["labels"]]
    frame_counts = Counter()

    for row in detection_rows:
        for label in set(row["labels"]):   # count per frame, not per detection
            frame_counts[label] += 1

    per_object = {
        label: {
            "total_detections": count,
            "frames_present":   frame_counts[label],
            "presence_pct":     round(frame_counts[label] / total * 100, 1),
        }
        for label, count in Counter(all_labels).items()
    }

    return {
        "most_common": Counter(all_labels).most_common(5),
        "persistent":  [l for l, s in per_object.items() if s["presence_pct"] > 50],
        "sporadic":    [l for l, s in per_object.items() if s["presence_pct"] < 10],
        "per_object":  per_object,
    }


def get_event_durations(detection_rows: list[dict], target_label: str, fps: float = 30) -> list[dict]:
    """
    Track continuous runs of a specific object being present.
    Useful for: 'how long was a person on screen?'

    Returns list of events with start/end times in seconds.
    """
    events  = []
    in_event = False
    start    = None

    for row in detection_rows:
        present = target_label in row["labels"]

        if present and not in_event:
            in_event = True
            start    = row["frame"]
        elif not present and in_event:
            in_event = False
            events.append({
                "label":      target_label,
                "start_sec":  round(start / fps, 2),
                "end_sec":    round(row["frame"] / fps, 2),
                "duration_sec": round((row["frame"] - start) / fps, 2),
            })

    if in_event:  # close last event
        events.append({
            "label":        target_label,
            "start_sec":    round(start / fps, 2),
            "end_sec":      round(detection_rows[-1]["frame"] / fps, 2),
            "duration_sec": round((detection_rows[-1]["frame"] - start) / fps, 2),
        })

    return events


def print_temporal_report(transitions: list[dict], object_stats: dict):
    """Pretty-print the full temporal EDA report."""

    print("\n--- Scene Timeline ---")
    for t in transitions:
        print(f"  Frames {t['start_frame']:>5}–{t['end_frame']:<5} │ {t['scene']}")

    print("\n--- Object Persistence ---")
    for label, stats in object_stats.get("per_object", {}).items():
        bar = "█" * int(stats["presence_pct"] / 5)  # 1 block per 5%
        print(f"  {label:<20} {stats['presence_pct']:>5.1f}%  {bar}")

    persistent = object_stats.get("persistent", [])
    if persistent:
        print(f"\n  Always present:  {', '.join(persistent)}")

    sporadic = object_stats.get("sporadic", [])
    if sporadic:
        print(f"  Briefly seen:    {', '.join(sporadic)}")