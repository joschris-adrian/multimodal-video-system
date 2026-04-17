from collections import Counter


# helpers 

SUBJECT_LABELS = {"person", "man", "woman", "child", "athlete", "player"}
ENV_MAP = {
    "street":  "an urban street",
    "indoor":  "an indoor setting",
    "beach":   "a beach",
    "sports":  "a sports venue",
    "nature":  "a natural environment",
    "vehicle": "a vehicle environment",
    "unknown": "an unidentified environment",
}

ACTION_KEYWORDS = {
    "walking":  ["walk", "strolling", "moving"],
    "running":  ["run", "sprint", "jog"],
    "talking":  ["talk", "speak", "say", "tell", "discuss"],
    "playing":  ["play", "playing", "game"],
    "working":  ["work", "working", "office", "desk"],
    "dancing":  ["danc", "dancing"],
    "swimming": ["swim", "swimming"],
}


def _extract_subject(objects: list[str]) -> str:
    for obj in objects:
        if obj in SUBJECT_LABELS:
            return "A person"
    return "The scene"


def _extract_action(transcript: str) -> str | None:
    t = transcript.lower()
    for action, keywords in ACTION_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return action
    return None


def _extract_environment(scene: str, objects: list[str]) -> str:
    if scene and scene != "unknown":
        return ENV_MAP.get(scene, "an unidentified environment")

    # fallback: infer from objects
    if any(o in objects for o in ["car", "truck", "bus", "bicycle"]):
        return ENV_MAP["street"]
    if any(o in objects for o in ["chair", "couch", "tv", "laptop"]):
        return ENV_MAP["indoor"]

    return ENV_MAP["unknown"]


# main function 

def generate_summary(
    video_name:     str,
    detection_rows: list[dict],
    transcript:     str,
    scene_rows:     list[dict] | None = None,
    transitions:    list[dict] | None = None,
    object_stats:   dict | None = None,
) -> str:

    # --- collect data ---
    all_labels = [l for r in detection_rows for l in r["labels"]]
    top3       = [obj for obj, _ in Counter(all_labels).most_common(3)]
    empty_pct  = (
        sum(1 for r in detection_rows if r["num_objects"] == 0)
        / len(detection_rows) * 100
        if detection_rows else 100
    )

    dominant_scene = "unknown"
    if scene_rows:
        scene_counts   = Counter(r["scene"] for r in scene_rows)
        dominant_scene = scene_counts.most_common(1)[0][0]

    # --- structured reasoning ---
    subject     = _extract_subject(top3)
    environment = _extract_environment(dominant_scene, top3)
    action      = _extract_action(transcript)

    if action:
        one_liner = f"{subject} is {action} in {environment}."
    elif top3:
        one_liner = f"{subject} appears in {environment}."
    else:
        one_liner = "No clear subject or action detected."

    # --- scene timeline ---
    timeline_str = ""
    if transitions:
        lines = [
            f"  {t['start_frame']}–{t['end_frame']}: {t['scene']}"
            for t in transitions
        ]
        timeline_str = "Scene timeline:\n" + "\n".join(lines) + "\n"

    # --- persistent objects ---
    persistent_str = ""
    if object_stats and object_stats.get("persistent"):
        persistent_str = f"Always present: {', '.join(object_stats['persistent'])}\n"

    # --- build full summary ---
    summary = (
        f"FILE: {video_name}\n"
        f"Summary: {one_liner}\n"
        f"Detected: {', '.join(top3) if top3 else 'nothing'}\n"
        f"Scene: {dominant_scene}\n"
        f"Audio: {transcript if transcript.strip() else 'No speech detected'}\n"
        f"Empty frames: {empty_pct:.1f}%\n"
        f"{timeline_str}"
        f"{persistent_str}"
    )

    return summary