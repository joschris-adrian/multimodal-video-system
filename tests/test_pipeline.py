import pytest
import os
import numpy as np
import pandas as pd
import cv2

#  helpers 

SAMPLE_VIDEO = "tests/assets/sample.mp4"  # tiny test video (see fixture below)

@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory):
    """Create a tiny synthetic video for testing — no real video file needed."""
    out_dir = tmp_path_factory.mktemp("assets")
    path    = str(out_dir / "sample.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30, (320, 240))

    for _ in range(60):  # 2 seconds @ 30fps
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


#  file_utils 

def test_list_videos_returns_only_video_files(tmp_path):
    from src.utils.file_utils import list_videos

    (tmp_path / "clip.mp4").touch()
    (tmp_path / "clip.avi").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "image.png").touch()

    result = list_videos(str(tmp_path))
    names  = [os.path.basename(p) for p in result]

    assert "clip.mp4" in names
    assert "clip.avi" in names
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


#  fusion / summarize 

def test_generate_summary_with_speech():
    from src.fusion.summarize import generate_summary

    detections = [
        {"labels": ["person", "car"], "num_objects": 2},
        {"labels": ["person"],        "num_objects": 1},
    ]
    summary = generate_summary("clip.mp4", detections, "I am walking down the street")

    assert "person" in summary
    assert "walking" in summary
    assert "clip.mp4" in summary


def test_generate_summary_no_speech():
    from src.fusion.summarize import generate_summary

    detections = [{"labels": ["dog"], "num_objects": 1}]
    summary    = generate_summary("clip.mp4", detections, "")

    assert "No speech" in summary


def test_generate_summary_no_detections():
    from src.fusion.summarize import generate_summary

    summary = generate_summary("clip.mp4", [], "")
    assert "No objects" in summary


def test_generate_summary_empty_pct():
    from src.fusion.summarize import generate_summary

    detections = [
        {"labels": [],          "num_objects": 0},
        {"labels": ["person"],  "num_objects": 1},
    ]
    summary = generate_summary("clip.mp4", detections, "")
    assert "50.0%" in summary


# ── transcribe (unit — no model loaded) ──────────────────────────────────────

def test_transcribe_quality_flags():
    """Test quality labelling logic directly without loading Whisper."""
    from src.audio.transcribe import _quality_label

    assert _quality_label("",                        None)   == "empty"
    assert _quality_label("hi",                      None)   == "too_short"
    assert _quality_label("word " * 10,              -1.5)   == "noisy"
    assert _quality_label("This is a proper sentence.", -0.3) == "good"