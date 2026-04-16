import whisper
import os
import pandas as pd
from moviepy import VideoFileClip
import numpy as np
import scipy.io.wavfile as wav

model = whisper.load_model("base")

def extract_audio_array(video_path, sr=16000):
    """Extract audio as numpy array via moviepy. No ffmpeg binary needed."""
    tmp = "temp/tmp_audio.wav"
    os.makedirs("temp", exist_ok=True)

    with VideoFileClip(video_path) as clip:
        if clip.audio is None:
            return None
        clip.audio.write_audiofile(tmp, fps=sr, logger=None)

    rate, data = wav.read(tmp)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32) / 32768.0

def _quality_label(text, avg_logprob):
    """Extracted for testability."""
    if not text.strip():               return "empty"
    if len(text.split()) < 5:         return "too_short"
    if avg_logprob and avg_logprob < -1.0: return "noisy"
    return "good"

def transcribe_video(video_path):
    """Transcribe a single video. Returns dict with transcript + quality info."""
    audio = extract_audio_array(video_path)

    if audio is None:
        return {"has_speech": False, "word_count": 0, "transcript": "", "quality": "no_audio"}

    result   = model.transcribe(audio, verbose=False)
    text     = result["text"].strip()
    segments = result.get("segments", [])
    num_segs = len(segments)
    avg_conf = sum(s["avg_logprob"] for s in segments) / num_segs if num_segs else None

    quality = _quality_label(text, avg_conf)

    return {
        "has_speech":   len(text) > 0,
        "word_count":   len(text.split()) if text else 0,
        "num_segments": num_segs,
        "avg_logprob":  round(avg_conf, 3) if avg_conf else None,
        "transcript":   text,
        "quality":      quality,
    }


def transcribe_folder(video_dir):
    """Run transcription on all videos in a folder. Returns a DataFrame."""
    video_exts = {".mp4", ".mov", ".avi", ".mkv"}
    rows = []

    for fname in os.listdir(video_dir):
        if os.path.splitext(fname)[1].lower() not in video_exts:
            continue
        path = os.path.join(video_dir, fname)
        print(f"Transcribing {fname}...")
        try:
            result = transcribe_video(path)
            rows.append({"file": fname, **result})
        except Exception as e:
            rows.append({"file": fname, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv("outputs/whisper_report.csv", index=False)
    print(f"\nSaved → outputs/whisper_report.csv")
    return df

