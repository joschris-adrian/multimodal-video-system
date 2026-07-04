# tests/test_video_llm.py

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import os
import tempfile


# ── RAM detection ────────────────────────────────────────────────────────────

def test_get_available_ram_gb_returns_float():
    from src.fusion.video_llm import _get_available_ram_gb
    result = _get_available_ram_gb()
    assert isinstance(result, float)
    assert result >= 0.0


def test_check_ram_returns_bool():
    from src.fusion.video_llm import _check_ram
    result = _check_ram("qwen2-vl-2b")
    assert isinstance(result, bool)


def test_check_ram_7b_stricter():
    """7B model should have higher RAM threshold than 2B."""
    from src.fusion.video_llm import _MIN_RAM_GB_2B, _MIN_RAM_GB_7B
    assert _MIN_RAM_GB_7B > _MIN_RAM_GB_2B


# ── Unload whisper helper ────────────────────────────────────────────────────

def test_unload_whisper_no_error():
    """Should not raise even if transcribe module has issues."""
    from src.fusion.video_llm import unload_whisper
    unload_whisper()  # should not raise


def test_unload_whisper_calls_transcribe_unload():
    from src.fusion.video_llm import unload_whisper
    with patch("src.fusion.video_llm.unload_whisper") as mock_unload:
        # Can't easily mock the internal import, but verify it doesn't crash
        pass
    # The function is a best-effort helper — just verify no exception
    unload_whisper()


# ── Image loading ────────────────────────────────────────────────────────────

def test_load_frame_images_empty_list():
    from src.fusion.video_llm import load_frame_images
    result = load_frame_images([])
    assert result == []


def test_load_frame_images_nonexistent_files():
    from src.fusion.video_llm import load_frame_images
    result = load_frame_images(["/nonexistent/frame.jpg"])
    assert result == []


def test_load_frame_images_valid_files():
    from src.fusion.video_llm import load_frame_images

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(3):
            path = os.path.join(tmpdir, f"frame_{i}.jpg")
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(path)
            paths.append(path)

        result = load_frame_images(paths)
        assert len(result) == 3
        assert all(isinstance(img, Image.Image) for img in result)
        assert all(img.mode == "RGB" for img in result)


def test_load_frame_images_respects_max_frames():
    """Default max_frames is 4 for memory safety."""
    from src.fusion.video_llm import load_frame_images, _DEFAULT_MAX_FRAMES

    # Verify the default is 4, not 8
    assert _DEFAULT_MAX_FRAMES == 4

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(10):
            path = os.path.join(tmpdir, f"frame_{i}.jpg")
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(path)
            paths.append(path)

        result = load_frame_images(paths, max_frames=4)
        assert len(result) == 4


def test_load_frame_images_resizes_large_images():
    """Default max_dim is 384 for memory safety."""
    from src.fusion.video_llm import load_frame_images, _IMAGE_MAX_DIM

    # Verify the default is 384, not 512
    assert _IMAGE_MAX_DIM == 384

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "large.jpg")
        img = Image.fromarray(np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8))
        img.save(path)

        result = load_frame_images([path])
        assert len(result) == 1
        assert max(result[0].size) <= 384


def test_load_frame_images_custom_max_dim():
    from src.fusion.video_llm import load_frame_images

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "large.jpg")
        img = Image.fromarray(np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8))
        img.save(path)

        result = load_frame_images([path], max_dim=200)
        assert max(result[0].size) <= 200


# ── Context builders ─────────────────────────────────────────────────────────

def test_detections_text_empty():
    from src.fusion.video_llm import _detections_text
    result = _detections_text([])
    assert result == "No objects detected."


def test_detections_text_with_data():
    from src.fusion.video_llm import _detections_text
    rows = [
        {"labels": ["person", "car"], "num_objects": 2},
        {"labels": ["person"], "num_objects": 1},
        {"labels": ["person", "car", "dog"], "num_objects": 3},
    ]
    result = _detections_text(rows)
    assert "person" in result
    assert "car" in result
    assert "dog" in result


def test_scene_text_empty():
    from src.fusion.video_llm import _scene_text
    result = _scene_text([], [])
    assert result == "No scene classification available."


def test_scene_text_single_scene():
    from src.fusion.video_llm import _scene_text
    rows = [
        {"frame": 0, "scene": "street"},
        {"frame": 30, "scene": "street"},
    ]
    result = _scene_text(rows, [])
    assert "street" in result
    assert "2/2" in result


def test_scene_text_multiple_transitions():
    from src.fusion.video_llm import _scene_text
    rows = [
        {"frame": 0, "scene": "street"},
        {"frame": 30, "scene": "indoor"},
        {"frame": 60, "scene": "beach"},
    ]
    transitions = [
        {"start_frame": 0, "end_frame": 30, "scene": "street"},
        {"start_frame": 30, "end_frame": 60, "scene": "indoor"},
        {"start_frame": 60, "end_frame": 90, "scene": "beach"},
    ]
    result = _scene_text(rows, transitions)
    assert "progression" in result
    assert "street → indoor → beach" in result


def test_nlp_text_empty():
    from src.fusion.video_llm import _nlp_text
    result = _nlp_text({})
    assert result == ""


def test_nlp_text_full():
    from src.fusion.video_llm import _nlp_text
    nlp = {
        "keywords": ["walking", "street"],
        "sentiment": {"label": "positive"},
        "topic": {"words": ["street", "car", "road", "urban", "person"]},
    }
    result = _nlp_text(nlp)
    assert "walking" in result
    assert "positive" in result


# ── Response parsing ─────────────────────────────────────────────────────────

def test_parse_response_standard_format():
    from src.fusion.video_llm import _parse_response
    text = """SUMMARY: A person is walking down a busy urban street.
DETAILS: They appear to be carrying a bag and moving quickly past several cars.
SCENE: street"""

    result = _parse_response(text)
    assert result["summary"] == "A person is walking down a busy urban street."
    assert "carrying a bag" in result["details"]
    assert result["scene"] == "street"


def test_parse_response_partial():
    from src.fusion.video_llm import _parse_response
    text = "SUMMARY: A dog runs through a park."

    result = _parse_response(text)
    assert result["summary"] == "A dog runs through a park."
    assert result["details"] == ""
    assert result["scene"] == "unknown"


def test_parse_response_empty():
    from src.fusion.video_llm import _parse_response
    result = _parse_response("")
    assert result["summary"] == "Video-LLM could not generate a summary."
    assert result["scene"] == "unknown"


def test_parse_response_unstructured():
    from src.fusion.video_llm import _parse_response
    text = "A person is surfing on a wave at sunset."

    result = _parse_response(text)
    assert result["summary"] == "A person is surfing on a wave at sunset."


def test_parse_response_case_insensitive():
    from src.fusion.video_llm import _parse_response
    text = "summary: Lower case summary line\nSCENE: BEACH"

    result = _parse_response(text)
    assert result["summary"] == "Lower case summary line"
    assert result["scene"] == "beach"


# ── Model info ───────────────────────────────────────────────────────────────

def test_get_model_info_returns_dict():
    from src.fusion.video_llm import get_model_info
    result = get_model_info("qwen2-vl-2b")
    assert isinstance(result, dict)
    assert "id" in result
    assert "key" in result
    assert "available" in result
    assert "size_mb" in result


def test_get_model_info_valid_key():
    from src.fusion.video_llm import get_model_info
    result = get_model_info("qwen2-vl-2b")
    assert result["key"] == "qwen2-vl-2b"
    assert "Qwen" in result["id"]


def test_get_model_info_custom_model_id():
    from src.fusion.video_llm import get_model_info
    result = get_model_info("some/custom-model")
    assert result["id"] == "some/custom-model"


# ── Model options ────────────────────────────────────────────────────────────

def test_model_options_has_expected_keys():
    from src.fusion.video_llm import MODEL_OPTIONS
    assert "qwen2-vl-2b" in MODEL_OPTIONS
    assert "qwen2-vl-7b" in MODEL_OPTIONS


def test_model_options_values_are_strings():
    from src.fusion.video_llm import MODEL_OPTIONS
    for key, value in MODEL_OPTIONS.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        assert "Qwen" in value


# ── Memory defaults ──────────────────────────────────────────────────────────

def test_memory_defaults_are_conservative():
    """Verify 8 GB RAM defaults are set correctly."""
    from src.fusion.video_llm import (
        _DEFAULT_MAX_FRAMES,
        _DEFAULT_MAX_TOKENS,
        _IMAGE_MAX_DIM,
        _MIN_RAM_GB_2B,
    )
    assert _DEFAULT_MAX_FRAMES == 4    # not 8
    assert _DEFAULT_MAX_TOKENS == 128  # not 256
    assert _IMAGE_MAX_DIM == 384       # not 512
    assert _MIN_RAM_GB_2B == 5.0       # conservative threshold


# ── Unload model ────────────────────────────────────────────────────────────

def test_unload_model_no_error_when_not_loaded():
    from src.fusion.video_llm import unload_model
    unload_model()  # should not raise
    unload_model()  # should not raise even twice


# ── Fusion dispatcher (integration with summarize.py) ───────────────────────

def test_fuse_dispatches_to_rule_based():
    from src.fusion.summarize import fuse

    result = fuse(
        video_name="test.mp4",
        detection_rows=[{"labels": ["person"], "num_objects": 1}],
        transcript="",
        mode="rule-based",
    )
    assert "FILE: test.mp4" in result
    assert "Fusion mode:" not in result


def test_fuse_rejects_invalid_mode():
    from src.fusion.summarize import fuse

    with pytest.raises(ValueError, match="Unknown fusion mode"):
        fuse(
            video_name="test.mp4",
            detection_rows=[],
            transcript="",
            mode="invalid-mode",
        )


def test_fuse_video_llm_falls_back_without_frames():
    """Video-LLM mode should fall back to rule-based if no frame_paths."""
    from src.fusion.summarize import fuse

    result = fuse(
        video_name="test.mp4",
        detection_rows=[{"labels": ["person"], "num_objects": 1}],
        transcript="",
        frame_paths=None,
        mode="video-llm",
    )
    assert "FILE: test.mp4" in result
    assert "video-llm" not in result.lower()


def test_fuse_video_llm_falls_back_on_oom():
    """Video-LLM mode should fall back on RuntimeError with OOM."""
    from src.fusion.summarize import fuse

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "frame.jpg")
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(path)

        with patch(
            "src.fusion.video_llm.generate_llm_summary",
            side_effect=RuntimeError("out of memory"),
        ):
            result = fuse(
                video_name="test.mp4",
                detection_rows=[{"labels": ["person"], "num_objects": 1}],
                transcript="",
                frame_paths=[path],
                mode="video-llm",
            )

    assert "FILE: test.mp4" in result
    assert "video-llm" not in result.lower()


def test_fuse_video_llm_falls_back_on_insufficient_ram():
    """Video-LLM mode should fall back on insufficient RAM error."""
    from src.fusion.summarize import fuse

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "frame.jpg")
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(path)

        with patch(
            "src.fusion.video_llm.generate_llm_summary",
            side_effect=RuntimeError("Insufficient RAM for Qwen"),
        ):
            result = fuse(
                video_name="test.mp4",
                detection_rows=[{"labels": ["person"], "num_objects": 1}],
                transcript="",
                frame_paths=[path],
                mode="video-llm",
            )

    assert "FILE: test.mp4" in result


def test_fuse_video_llm_with_mock():
    """Video-LLM mode should return LLM summary when successful."""
    from src.fusion.summarize import fuse

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "frame.jpg")
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(path)

        mock_summary = (
            "FILE: test.mp4\n"
            "Summary:  A person is surfing on a wave at sunset.\n"
            "Detected: person, surfboard\n"
            "Scene:    beach\n"
            "Audio:    No speech detected\n"
            "Empty frames: 0.0%\n"
            "Fusion mode: video-llm (Qwen/Qwen2-VL-2B-Instruct)\n"
        )

        with patch(
            "src.fusion.video_llm.generate_llm_summary",
            return_value=mock_summary,
        ):
            result = fuse(
                video_name="test.mp4",
                detection_rows=[{"labels": ["person", "surfboard"], "num_objects": 2}],
                transcript="",
                frame_paths=[path],
                mode="video-llm",
            )

    assert "video-llm" in result
    assert "surfing on a wave" in result


# ── Prompt building ──────────────────────────────────────────────────────────

def test_build_messages_includes_system_prompt():
    from src.fusion.video_llm import _build_messages

    images = [Image.new("RGB", (100, 100))]
    messages = _build_messages(
        images=images,
        transcript="hello world",
        detections_summary="person (5 frames)",
        scene_summary="street",
        nlp_summary="",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "expert video analyst" in messages[0]["content"]


def test_build_messages_user_has_images_and_text():
    from src.fusion.video_llm import _build_messages

    images = [Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100))]
    messages = _build_messages(
        images=images,
        transcript="test transcript",
        detections_summary="car (3 frames)",
        scene_summary="street",
        nlp_summary="Keywords: car",
    )

    user_content = messages[1]["content"]
    assert len(user_content) == 3  # 2 images + 1 text
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "image"
    assert user_content[2]["type"] == "text"


def test_build_messages_no_images():
    from src.fusion.video_llm import _build_messages

    messages = _build_messages(
        images=[],
        transcript="test",
        detections_summary="none",
        scene_summary="unknown",
        nlp_summary="",
    )

    user_content = messages[1]["content"]
    assert len(user_content) == 1
    assert user_content[0]["type"] == "text"


# ── Transcribe unload/reload ─────────────────────────────────────────────────

def test_transcribe_unload_model():
    from src.audio.transcribe import unload_model, _ensure_model

    # Unload should set model to None
    unload_model()
    # _ensure_model should reload it
    m = _ensure_model()
    assert m is not None


def test_transcribe_still_works_after_unload():
    """Verify transcription works after unload/reload cycle."""
    from src.audio.transcribe import unload_model, transcribe_video

    unload_model()

    # Create a tiny synthetic video for testing
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, 1, (64, 64))
        for _ in range(5):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # This will return "no_audio" since synthetic video has no audio
        result = transcribe_video(tmp_path)
        assert "quality" in result
        assert "transcript" in result
    finally:
        os.unlink(tmp_path)