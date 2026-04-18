
def test_build_prompt_person_detected():
    from src.generation.generate_image import build_prompt
    prompt = build_prompt(["person", "car"], "street", "")
    assert "person" in prompt.lower() or "A person" in prompt


def test_build_prompt_no_person():
    from src.generation.generate_image import build_prompt
    prompt = build_prompt(["car", "truck"], "street", "")
    assert "scene" in prompt.lower()


def test_build_prompt_action_from_transcript():
    from src.generation.generate_image import build_prompt
    prompt = build_prompt(["person"], "indoor", "She was walking down the road")
    assert "walking" in prompt


def test_build_prompt_action_from_objects():
    from src.generation.generate_image import build_prompt
    # no transcript — action should come from detected objects
    prompt = build_prompt(["person", "surfboard"], "sports", "")
    assert "surfing" in prompt


def test_build_prompt_haircut_from_objects():
    from src.generation.generate_image import build_prompt
    prompt = build_prompt(["person", "hair drier"], "indoor", "")
    assert "blowdry" in prompt or "hair" in prompt


def test_build_prompt_action_from_transcript_beats_objects():
    from src.generation.generate_image import build_prompt
    # transcript should take priority over object hint
    prompt = build_prompt(["person", "surfboard"], "sports", "She was running fast")
    assert "running" in prompt


def test_build_prompt_scene_street():
    from src.generation.generate_image import build_prompt
    prompt = build_prompt(["person"], "street", "")
    assert "street" in prompt or "urban" in prompt


def test_build_prompt_scene_beach():
    from src.generation.generate_image import build_prompt
    prompt = build_prompt(["person"], "beach", "")
    assert "beach" in prompt


def test_build_prompt_scene_unknown_fallback():
    from src.generation.generate_image import build_prompt
    prompt = build_prompt(["person"], "unknown", "")
    assert "environment" in prompt


def test_build_prompt_default_action_standing():
    from src.generation.generate_image import build_prompt
    # no transcript, no object hint → should default to standing
    prompt = build_prompt(["person"], "indoor", "")
    assert "standing" in prompt


def test_build_prompt_contains_quality_words():
    from src.generation.generate_image import build_prompt
    prompt = build_prompt(["person"], "street", "walking")
    assert "photorealistic" in prompt
    assert "lighting" in prompt


# tracking / SimpleTracker 

def test_tracker_assigns_id_to_first_detection():
    from src.tracking.track_objects import SimpleTracker
    tracker = SimpleTracker()
    dets    = [{"bbox": [10, 10, 50, 50], "label": "person", "conf": 0.9}]
    result  = tracker.update(dets)
    assert len(result) == 1
    assert "track_id" in result[0]


def test_tracker_same_box_keeps_id():
    from src.tracking.track_objects import SimpleTracker
    tracker = SimpleTracker()
    dets    = [{"bbox": [10, 10, 50, 50], "label": "person", "conf": 0.9}]

    first  = tracker.update(dets)
    second = tracker.update(dets)

    assert first[0]["track_id"] == second[0]["track_id"]


def test_tracker_different_box_new_id():
    from src.tracking.track_objects import SimpleTracker
    tracker = SimpleTracker()

    first  = tracker.update([{"bbox": [10, 10,  50,  50], "label": "person", "conf": 0.9}])
    second = tracker.update([{"bbox": [300, 300, 350, 350], "label": "person", "conf": 0.9}])

    # boxes don't overlap at all → different track IDs
    assert first[0]["track_id"] != second[0]["track_id"]


def test_tracker_multiple_objects():
    from src.tracking.track_objects import SimpleTracker
    tracker = SimpleTracker()
    dets = [
        {"bbox": [10,  10,  50,  50],  "label": "person", "conf": 0.9},
        {"bbox": [200, 200, 250, 250], "label": "car",    "conf": 0.8},
    ]
    result = tracker.update(dets)
    ids    = [r["track_id"] for r in result]
    assert len(set(ids)) == 2   # two distinct IDs


def test_tracker_empty_detections():
    from src.tracking.track_objects import SimpleTracker
    tracker = SimpleTracker()
    result  = tracker.update([])
    assert result == []


def test_tracker_lost_track_removed():
    from src.tracking.track_objects import SimpleTracker
    tracker = SimpleTracker(max_lost=2)

    tracker.update([{"bbox": [10, 10, 50, 50], "label": "person", "conf": 0.9}])

    # send empty detections max_lost+1 times → track should be dropped
    for _ in range(3):
        tracker.update([])

    assert len(tracker.tracks) == 0


def test_tracker_iou_computed_correctly():
    from src.tracking.track_objects import SimpleTracker
    tracker = SimpleTracker()

    # perfect overlap
    iou = tracker._iou([0, 0, 10, 10], [0, 0, 10, 10])
    assert iou == 1.0

    # no overlap
    iou = tracker._iou([0, 0, 10, 10], [20, 20, 30, 30])
    assert iou == 0.0

    # partial overlap
    iou = tracker._iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0.0 < iou < 1.0