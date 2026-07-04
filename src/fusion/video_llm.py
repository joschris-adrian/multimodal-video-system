"""
Video-LLM fusion module using Qwen2-VL for contextual video summarization.

Memory-optimised for 8 GB RAM systems:
  - Unloads Whisper before loading VLM (~150 MB freed)
  - Uses low_cpu_mem_usage=True to halve peak load memory
  - Limits to 4 frames at 384 px (vs 8 at 512 px)
  - Caps generation at 128 tokens (smaller KV cache)
  - Unloads VLM immediately after inference
"""

import gc
import logging
import platform
from typing import Optional
from collections import Counter

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Model configuration ──────────────────────────────────────────────────────

MODEL_OPTIONS = {
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
}

DEFAULT_MODEL_KEY = "qwen2-vl-2b"

# Memory-tuned defaults for 8 GB RAM
_DEFAULT_MAX_FRAMES = 4
_DEFAULT_MAX_TOKENS = 128
_IMAGE_MAX_DIM = 384

# Minimum free RAM (GB) to attempt loading — conservative
_MIN_RAM_GB_2B = 5.0
_MIN_RAM_GB_7B = 12.0

# Lazy-loaded globals
_model = None
_processor = None
_current_model_id: Optional[str] = None


# ── RAM detection ────────────────────────────────────────────────────────────

def _get_available_ram_gb() -> float:
    """
    Get approximate available RAM in GB.
    Returns 0.0 on unsupported platforms (triggers safety fallback).
    """
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if parts and parts[0] == "MemAvailable:":
                        return int(parts[1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _check_ram(model_key: str = DEFAULT_MODEL_KEY) -> bool:
    """
    Return True if there appears to be enough RAM.
    Logs a warning if RAM is low or undetectable.
    """
    available = _get_available_ram_gb()
    threshold = _MIN_RAM_GB_2B if "2b" in model_key else _MIN_RAM_GB_7B

    if available == 0.0:
        logger.warning(
            "Could not detect available RAM. "
            "Proceeding — may OOM on systems with <8 GB."
        )
        return True  # can't confirm, so allow attempt

    if available < threshold:
        logger.warning(
            f"Only {available:.1f} GB RAM available — "
            f"{threshold:.0f} GB recommended for {model_key}. "
            "Will attempt load but may fall back to rule-based fusion."
        )
        return False

    logger.info(f"RAM check passed: {available:.1f} GB available")
    return True


# ── Whisper unload helper ────────────────────────────────────────────────────

def unload_whisper():
    """
    Free the Whisper model that lives as a module-level global
    in src.audio.transcribe.  Safe to call multiple times.
    """
    try:
        from src.audio.transcribe import unload_model as unload_w
        unload_w()
        logger.info("Whisper model unloaded to free RAM")
    except Exception as e:
        logger.debug(f"Could not unload Whisper: {e}")


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(model_key: str = DEFAULT_MODEL_KEY, device: Optional[str] = None):
    """
    Load Qwen2-VL model and processor. Cached after first call.

    Uses low_cpu_mem_usage=True to reduce peak memory during loading.
    On CPU, uses float16 for weight storage (half the RAM).

    Args:
        model_key: Key from MODEL_OPTIONS or a full HuggingFace model ID.
        device: 'cuda', 'cpu', or None (auto-detect).

    Returns:
        Tuple of (model, processor).

    Raises:
        ImportError: If transformers is too old or missing dependencies.
        RuntimeError: If model fails to load or insufficient RAM.
    """
    global _model, _processor, _current_model_id

    model_id = MODEL_OPTIONS.get(model_key, model_key)

    # Return cached if same model
    if _model is not None and _current_model_id == model_id:
        return _model, _processor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # RAM check on CPU
    if device == "cpu" and not _check_ram(model_key):
        raise RuntimeError(
            f"Insufficient RAM for {model_id}. "
            "Use --fusion-mode rule-based or close other applications."
        )

    logger.info(f"Loading Video-LLM: {model_id} on {device}")

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "Video-LLM requires transformers >= 4.40 and qwen-vl-utils.\n"
            "Install with: pip install 'transformers>=4.40' qwen-vl-utils"
        ) from e

    try:
        # Force garbage collection before loading
        gc.collect()

        _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.float16 if device == "cuda" else torch.float16

        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _model = _model.to(device)
        _model.eval()

        # Free any temporary memory from loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except RuntimeError as e:
        _model = None
        _processor = None
        if "out of memory" in str(e).lower():
            raise RuntimeError(
                f"Out of memory loading {model_id}. "
                "Close other applications or use rule-based fusion."
            ) from e
        raise
    except Exception as e:
        _model = None
        _processor = None
        raise RuntimeError(f"Failed to load {model_id}: {e}") from e

    _current_model_id = model_id
    logger.info(f"Video-LLM loaded: {model_id}")
    return _model, _processor


def unload_model():
    """Free GPU/CPU memory by deleting the cached model."""
    global _model, _processor, _current_model_id
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    _current_model_id = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Video-LLM unloaded")


def is_available(model_key: str = DEFAULT_MODEL_KEY) -> bool:
    """Check if model can be downloaded (no full load)."""
    model_id = MODEL_OPTIONS.get(model_key, model_key)
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return True
    except Exception:
        return False


def get_model_info(model_key: str = DEFAULT_MODEL_KEY) -> dict:
    """Return metadata about a model without loading it."""
    model_id = MODEL_OPTIONS.get(model_key, model_key)
    info = {"id": model_id, "key": model_key, "available": False, "size_mb": 0}

    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        info["available"] = True
        hidden = getattr(config, "hidden_size", 0)
        layers = getattr(config, "num_hidden_layers", 0)
        vocab = getattr(config, "vocab_size", 0)
        if hidden and layers:
            embed_mb = vocab * hidden * 2 / 1024 / 1024
            per_layer_mb = hidden * hidden * 12 * 2 / 1024 / 1024
            info["size_mb"] = int(embed_mb + per_layer_mb * layers)
    except Exception:
        pass

    return info


# ── Image loading ────────────────────────────────────────────────────────────

def load_frame_images(
    frame_paths: list[str],
    max_frames: int = _DEFAULT_MAX_FRAMES,
    max_dim: int = _IMAGE_MAX_DIM,
) -> list[Image.Image]:
    """
    Load frame images, sampling evenly if too many.
    Images are resized to max_dim to limit VLM input memory.

    Args:
        frame_paths: List of file paths to frame JPGs.
        max_frames: Maximum images to return (default 4 for 8 GB RAM).
        max_dim: Maximum pixel dimension per image (default 384).

    Returns:
        List of PIL Images in RGB mode.
    """
    if not frame_paths:
        return []

    if len(frame_paths) > max_frames:
        step = len(frame_paths) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        frame_paths = [frame_paths[i] for i in indices]

    images = []
    for path in frame_paths:
        try:
            img = Image.open(path).convert("RGB")
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            images.append(img)
        except Exception as e:
            logger.warning(f"Could not load frame {path}: {e}")

    return images


# ── Context builders ─────────────────────────────────────────────────────────

def _detections_text(detection_rows: list[dict]) -> str:
    """Summarise detection results for the LLM prompt."""
    if not detection_rows:
        return "No objects detected."
    all_labels = [l for r in detection_rows for l in r["labels"]]
    counts = Counter(all_labels)
    parts = [f"{label} ({count} frames)" for label, count in counts.most_common(10)]
    return "Detected objects: " + ", ".join(parts)


def _scene_text(scene_rows: list[dict], transitions: list[dict]) -> str:
    """Summarise scene classification for the LLM prompt."""
    if not scene_rows:
        return "No scene classification available."
    counts = Counter(r["scene"] for r in scene_rows)
    dominant, count = counts.most_common(1)[0]
    result = f"Dominant scene: {dominant} ({count}/{len(scene_rows)} frames)"
    if transitions and len(transitions) > 1:
        seq = " → ".join(t["scene"] for t in transitions)
        result += f"\nScene progression: {seq}"
    return result


def _nlp_text(nlp: dict) -> str:
    """Summarise NLP analysis for the LLM prompt."""
    parts = []
    if nlp.get("keywords"):
        parts.append(f"Keywords: {', '.join(nlp['keywords'])}")
    if nlp.get("sentiment"):
        parts.append(f"Sentiment: {nlp['sentiment']['label']}")
    if nlp.get("topic") and nlp["topic"].get("words"):
        parts.append(f"Topic words: {', '.join(nlp['topic']['words'][:5])}")
    return " | ".join(parts) if parts else ""


# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert video analyst. You will receive:
- A sequence of frames sampled from a video
- The speech transcript (if any)
- Object detection statistics
- Scene classification results

Write a concise, factual summary of what is happening.

RULES:
- Start with one clear sentence describing the main subject, action, and setting
- Be specific about actions (e.g. "surfing on a wave" not just "in water")
- Mention key objects only if they help understand the scene
- If there is useful speech, incorporate it naturally
- For multi-scene videos, briefly note the transition
- 2-4 sentences total, no bullet points

RESPOND IN THIS EXACT FORMAT:
SUMMARY: <one sentence main summary>
DETAILS: <1-3 sentences of additional context>
SCENE: <single dominant scene label>"""

USER_TEMPLATE = """\
Transcript: {transcript}

{detections}

{scene}
{nlp}

Describe what is happening in this video."""


def _build_messages(
    images: list[Image.Image],
    transcript: str,
    detections_summary: str,
    scene_summary: str,
    nlp_summary: str,
) -> list[dict]:
    """Build the chat messages for the model."""
    user_text = USER_TEMPLATE.format(
        transcript=transcript.strip() or "[No speech detected]",
        detections=detections_summary,
        scene=scene_summary,
        nlp=f"\n{nlp_summary}" if nlp_summary else "",
    )

    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# ── Response parsing ─────────────────────────────────────────────────────────

def _parse_response(text: str) -> dict:
    """
    Parse structured LLM response into components.

    Returns:
        dict with keys: summary, details, scene
    """
    result = {"summary": "", "details": "", "scene": "unknown"}

    for line in text.strip().split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("SUMMARY:"):
            result["summary"] = line[len("SUMMARY:"):].strip()
        elif upper.startswith("DETAILS:"):
            result["details"] = line[len("DETAILS:"):].strip()
        elif upper.startswith("SCENE:"):
            result["scene"] = line[len("SCENE:"):].strip().lower()

    if not result["summary"]:
        cleaned = text.strip().strip('"').strip("'")
        if cleaned:
            result["summary"] = cleaned
        else:
            result["summary"] = "Video-LLM could not generate a summary."

    return result


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_llm_summary(
    video_name: str,
    detection_rows: list[dict],
    transcript: str,
    frame_paths: list[str],
    scene_rows: list[dict] | None = None,
    transitions: list[dict] | None = None,
    object_stats: dict | None = None,
    nlp: dict | None = None,
    model_key: str = DEFAULT_MODEL_KEY,
    max_frames: int = _DEFAULT_MAX_FRAMES,
    max_new_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """
    Generate a video summary using Qwen2-VL.

    Memory-optimised for 8 GB RAM:
      - Unloads Whisper model before loading VLM
      - Uses fp16 + low_cpu_mem_usage to halve weight memory
      - Limits frames and generation length to reduce KV cache
      - Unloads VLM immediately after inference

    Drop-in alternative to generate_summary() from summarize.py.
    Produces output in the same format so downstream code works unchanged.

    Args:
        video_name: Name of the video file.
        detection_rows: List of detection dicts from YOLOv8.
        transcript: Whisper transcript text.
        frame_paths: Paths to sampled frame JPGs.
        scene_rows: Scene classification results.
        transitions: Scene transition data from aggregator.
        object_stats: Object persistence statistics.
        nlp: NLP analysis results.
        model_key: Key from MODEL_OPTIONS or full HuggingFace ID.
        max_frames: Max frames to send (default 4 for 8 GB).
        max_new_tokens: Max tokens to generate (default 128 for 8 GB).

    Returns:
        Formatted summary string (same format as rule-based fusion).
    """
    # Free Whisper memory before loading VLM
    unload_whisper()
    gc.collect()

    # Load model
    model, processor = load_model(model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load images (small and few for memory)
    images = load_frame_images(frame_paths, max_frames=max_frames)
    if not images:
        logger.warning("No frames available — cannot run Video-LLM without images")

    # Build context
    det_text = _detections_text(detection_rows)
    scene_text = _scene_text(scene_rows or [], transitions or [])
    nlp_text = _nlp_text(nlp) if nlp else ""

    # Build messages
    messages = _build_messages(images, transcript, det_text, scene_text, nlp_text)

    # ── Inference ─────────────────────────────────────────────────────────
    raw_response = ""
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # inference_mode is more memory-efficient than no_grad
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_response = processor.batch_decode(new_ids, skip_special_tokens=True)[0]

        # Free inference tensors immediately
        del inputs, output_ids, new_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"Video-LLM OOM during inference: {e}")
        else:
            logger.error(f"Video-LLM inference failed: {e}")
        raw_response = ""
    except Exception as e:
        logger.error(f"Video-LLM inference failed: {e}")
        raw_response = ""
    finally:
        # Always unload VLM immediately — it's too large to keep resident
        unload_model()

    # Parse
    parsed = _parse_response(raw_response)

    # ── Build output (same format as rule-based) ──────────────────────────
    all_labels = [l for r in detection_rows for l in r["labels"]]
    top3 = [obj for obj, _ in Counter(all_labels).most_common(3)]

    empty_pct = (
        sum(1 for r in detection_rows if r["num_objects"] == 0)
        / len(detection_rows) * 100
        if detection_rows else 100
    )

    dominant_scene = "unknown"
    if scene_rows:
        scene_counts = Counter(r["scene"] for r in scene_rows)
        dominant_scene = scene_counts.most_common(1)[0][0]

    # Timeline
    timeline_str = ""
    if transitions:
        lines = [
            f"  {t['start_frame']}–{t['end_frame']}: {t['scene']}"
            for t in transitions
        ]
        timeline_str = "Scene timeline:\n" + "\n".join(lines) + "\n"

    # Persistent objects
    persistent_str = ""
    if object_stats and object_stats.get("persistent"):
        persistent_str = f"Always present: {', '.join(object_stats['persistent'])}\n"

    # NLP block
    nlp_str = ""
    if nlp:
        from src.fusion.summarize import SENTIMENT_EMOJI
        keywords = ", ".join(nlp["keywords"]) if nlp.get("keywords") else "none"
        sentiment = nlp["sentiment"]["label"]
        emoji = SENTIMENT_EMOJI.get(sentiment, "")
        nlp_str = f"Keywords:  {keywords}\nSentiment: {sentiment} {emoji}\n"
        if nlp.get("topic") and nlp["topic"].get("words"):
            topic_words = ", ".join(nlp["topic"]["words"][:4])
            nlp_str += f"Topic:     {topic_words}\n"

    one_liner = parsed["summary"]

    summary = (
        f"FILE: {video_name}\n"
        f"Summary:  {one_liner}\n"
        f"Detected: {', '.join(top3) if top3 else 'nothing'}\n"
        f"Scene:    {dominant_scene}\n"
        f"Audio:    {transcript if transcript.strip() else 'No speech detected'}\n"
        f"{nlp_str}"
        f"Empty frames: {empty_pct:.1f}%\n"
        f"{timeline_str}"
        f"{persistent_str}"
    )

    if parsed["details"]:
        summary += f"LLM Details: {parsed['details']}\n"

    model_id = MODEL_OPTIONS.get(model_key, model_key)
    summary += f"Fusion mode: video-llm ({model_id})\n"

    return summary