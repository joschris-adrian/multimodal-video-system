# tests/test_pipeline.py

import pytest
import os
import numpy as np
import cv2

#  fixture 

@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory):
    """Synthetic 2-second video — no real file needed."""
    out_dir = tmp_path_factory.mktemp("assets")
    path    = str(out_dir / "sample.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30, (320, 240))
    for _ in range(60):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def tiny_frame():
    """Single random BGR frame."""
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


#  file_utils 

def test_list_videos_returns_only_video_files(tmp_path):
    from src.utils.file_utils import list_videos
    (tmp_path / "clip.mp4").touch()
    (tmp_path / "clip.avi").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "image.png").touch()

    names = [os.path.basename(p) for p in list_videos(str(tmp_path))]
    assert "clip.mp4"  in names
    assert "clip.avi"  in names
    assert "notes.txt" not in names
    assert "image.png" not in names


def test_list_videos_empty_folder(tmp_path):
    from src.utils.file_utils import list_videos
    assert list_videos(str(tmp_path)) == []


def test_list_videos_missing_folder():
    from src.utils.file_utils import list_videos
    with pytest.raises(FileNotFoundError):
        list_videos("/does/not/exist")


def test_ensure_dirs_creates_folders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.utils.file_utils import ensure_dirs
    ensure_dirs()
    assert os.path.isdir("temp/frames")
    assert os.path.isdir("outputs/annotated_frames")


#  extract_frames 

def test_extract_frames_returns_list(tiny_video, tmp_path):
    from src.video.extract_frames import extract_frames
    result = extract_frames(tiny_video, str(tmp_path / "frames"), every_n=15)
    assert isinstance(result, list)
    assert len(result) > 0


def test_extract_frames_saves_jpgs(tiny_video, tmp_path):
    from src.video.extract_frames import extract_frames
    out = str(tmp_path / "frames")
    extract_frames(tiny_video, out, every_n=10)
    jpgs = [f for f in os.listdir(out) if f.endswith(".jpg")]
    assert len(jpgs) > 0


def test_extract_frames_metadata_keys(tiny_video, tmp_path):
    from src.video.extract_frames import extract_frames
    result = extract_frames(tiny_video, str(tmp_path / "frames"), every_n=15)
    for row in result:
        assert "frame_idx" in row
        assert "second"    in row
        assert "path"      in row


#  classify_scene 

def test_classify_frame_returns_expected_keys(tiny_frame):
    from src.vision.classify_scene import classify_frame
    result = classify_frame(tiny_frame)
    assert "scene"      in result
    assert "top_class"  in result
    assert "confidence" in result
    assert "top3"       in result


def test_classify_frame_scene_is_string(tiny_frame):
    from src.vision.classify_scene import classify_frame
    result = classify_frame(tiny_frame)
    assert isinstance(result["scene"], str)
    assert len(result["scene"]) > 0


def test_classify_frame_confidence_range(tiny_frame):
    from src.vision.classify_scene import classify_frame
    result = classify_frame(tiny_frame)
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_frame_top3_length(tiny_frame):
    from src.vision.classify_scene import classify_frame
    result = classify_frame(tiny_frame)
    assert len(result["top3"]) == 3


def test_classify_video_returns_list(tiny_video):
    from src.vision.classify_scene import classify_video
    result = classify_video(tiny_video, every_n=15)
    assert isinstance(result, list)
    assert len(result) > 0


def test_classify_video_row_keys(tiny_video):
    from src.vision.classify_scene import classify_video
    result = classify_video(tiny_video, every_n=15)
    for row in result:
        assert "frame"  in row
        assert "second" in row
        assert "scene"  in row


def test_map_to_scene_known_keywords():
    from src.vision.classify_scene import _map_to_scene
    assert _map_to_scene("seashore")    == "beach"
    assert _map_to_scene("street sign") == "street"
    assert _map_to_scene("living room") == "indoor"
    assert _map_to_scene("random xyz")  == "unknown"


#  segment_objects 

def test_segment_frame_returns_keys(tiny_frame):
    from src.vision.segment_objects import segment_frame
    result = segment_frame(tiny_frame)
    assert "annotated" in result
    assert "segments"  in result


def test_segment_frame_annotated_shape(tiny_frame):
    from src.vision.segment_objects import segment_frame
    result = segment_frame(tiny_frame)
    assert result["annotated"].shape == tiny_frame.shape


def test_segment_frame_segments_is_list(tiny_frame):
    from src.vision.segment_objects import segment_frame
    result = segment_frame(tiny_frame)
    assert isinstance(result["segments"], list)


def test_segment_frame_segment_keys(tiny_frame):
    """If any segments detected, check they have required keys."""
    from src.vision.segment_objects import segment_frame
    result = segment_frame(tiny_frame)
    for seg in result["segments"]:
        assert "label"      in seg
        assert "confidence" in seg
        assert "bbox"       in seg
        assert len(seg["bbox"]) == 4


def test_segment_video_saves_files(tiny_video, tmp_path):
    from src.vision.segment_objects import segment_video
    out = str(tmp_path / "seg_out")
    rows = segment_video(tiny_video, out, every_n=15)
    assert isinstance(rows, list)
    assert os.path.isdir(out)


#  temporal aggregator 

def test_get_scene_transitions_basic():
    from src.temporal.aggregator import get_scene_transitions
    rows = [
        {"frame": 0,  "scene": "street"},
        {"frame": 30, "scene": "street"},
        {"frame": 60, "scene": "indoor"},
        {"frame": 90, "scene": "indoor"},
    ]
    result = get_scene_transitions(rows, min_duration=1)
    scenes = [t["scene"] for t in result]
    assert "street" in scenes
    assert "indoor" in scenes


def test_get_scene_transitions_empty():
    from src.temporal.aggregator import get_scene_transitions
    assert get_scene_transitions([]) == []


def test_get_scene_transitions_filters_flickers():
    from src.temporal.aggregator import get_scene_transitions
    # single-frame flicker of "beach" should be filtered out
    rows = [
        {"frame": 0,  "scene": "street"},
        {"frame": 30, "scene": "beach"},   # flicker — only 1 frame
        {"frame": 60, "scene": "street"},
        {"frame": 90, "scene": "street"},
    ]
    result = get_scene_transitions(rows, min_duration=2)
    scenes = [t["scene"] for t in result]
    assert "beach" not in scenes


def test_get_object_stats_most_common():
    from src.temporal.aggregator import get_object_stats
    rows = [
        {"labels": ["person", "car"],  "num_objects": 2},
        {"labels": ["person"],         "num_objects": 1},
        {"labels": ["person", "dog"],  "num_objects": 2},
    ]
    stats = get_object_stats(rows)
    top   = [label for label, _ in stats["most_common"]]
    assert top[0] == "person"


def test_get_object_stats_persistent():
    from src.temporal.aggregator import get_object_stats
    rows = [{"labels": ["person"], "num_objects": 1}] * 10
    stats = get_object_stats(rows)
    assert "person" in stats["persistent"]


def test_get_object_stats_empty():
    from src.temporal.aggregator import get_object_stats
    assert get_object_stats([]) == {}


def test_get_object_stats_presence_pct():
    from src.temporal.aggregator import get_object_stats
    rows = [
        {"labels": ["person"], "num_objects": 1},
        {"labels": [],         "num_objects": 0},
    ]
    stats = get_object_stats(rows)
    assert stats["per_object"]["person"]["presence_pct"] == 50.0


def test_get_event_durations_detects_run():
    from src.temporal.aggregator import get_event_durations
    rows = [
        {"frame": 0,  "labels": ["person"]},
        {"frame": 30, "labels": ["person"]},
        {"frame": 60, "labels": []},
        {"frame": 90, "labels": ["person"]},
    ]
    events = get_event_durations(rows, "person", fps=30)
    assert len(events) == 2
    assert events[0]["label"] == "person"


def test_get_event_durations_no_target():
    from src.temporal.aggregator import get_event_durations
    rows = [{"frame": 0, "labels": ["car"]},
            {"frame": 30, "labels": ["car"]}]
    events = get_event_durations(rows, "person", fps=30)
    assert events == []


#  fusion / summarize 

def test_generate_summary_one_liner_with_action():
    from src.fusion.summarize import generate_summary
    detections = [{"labels": ["person"], "num_objects": 1}]
    scene_rows = [{"scene": "street"}]
    summary    = generate_summary("clip.mp4", detections, "I am walking down the road",
                                  scene_rows=scene_rows)
    assert "walking" in summary
    assert "street"  in summary


def test_generate_summary_no_speech():
    from src.fusion.summarize import generate_summary
    detections = [{"labels": ["dog"], "num_objects": 1}]
    summary    = generate_summary("clip.mp4", detections, "")
    assert "No speech" in summary


def test_generate_summary_no_detections():
    from src.fusion.summarize import generate_summary
    summary = generate_summary("clip.mp4", [], "")
    assert "No objects" in summary or "nothing" in summary


def test_generate_summary_empty_pct():
    from src.fusion.summarize import generate_summary
    detections = [
        {"labels": [],         "num_objects": 0},
        {"labels": ["person"], "num_objects": 1},
    ]
    summary = generate_summary("clip.mp4", detections, "")
    assert "50.0%" in summary


def test_generate_summary_includes_timeline():
    from src.fusion.summarize import generate_summary
    detections  = [{"labels": ["person"], "num_objects": 1}]
    transitions = [{"start_frame": 0, "end_frame": 60, "scene": "beach",
                    "duration_frames": 60}]
    summary = generate_summary("clip.mp4", detections, "", transitions=transitions)
    assert "beach" in summary
    assert "0" in summary


def test_generate_summary_persistent_objects():
    from src.fusion.summarize import generate_summary
    detections   = [{"labels": ["person"], "num_objects": 1}]
    object_stats = {"persistent": ["person"], "sporadic": [], "per_object": {}, "most_common": []}
    summary = generate_summary("clip.mp4", detections, "", object_stats=object_stats)
    assert "person" in summary


# transcribe helpers 

def test_quality_label_empty():
    from src.audio.transcribe import _quality_label
    assert _quality_label("", None) == "empty"


def test_quality_label_too_short():
    from src.audio.transcribe import _quality_label
    assert _quality_label("hi", None) == "too_short"


def test_quality_label_noisy():
    from src.audio.transcribe import _quality_label
    assert _quality_label("word " * 10, -1.5) == "noisy"


def test_quality_label_good():
    from src.audio.transcribe import _quality_label
    assert _quality_label("This is a proper sentence.", -0.3) == "good"