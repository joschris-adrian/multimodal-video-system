import argparse
import os
from src.utils.file_utils import list_videos, ensure_dirs
from src.video.extract_frames import extract_frames
from src.vision.detect_objects  import detect_objects
from src.vision.classify_scene  import classify_video
from src.vision.segment_objects import segment_video
from src.audio.transcribe       import transcribe_video
from src.audio.nlp_analysis     import (
    tfidf_keywords, train_word2vec, train_lda,
    analyze_transcript,
)
from src.temporal.aggregator    import (
    get_scene_transitions,
    get_object_stats,
    get_event_durations,
    print_temporal_report,
)
from src.fusion.summarize import fuse


def run(video_path, nlp_results=None, fusion_mode="rule-based", model_key="qwen2-vl-2b"):
    name = os.path.basename(video_path)
    stem = os.path.splitext(name)[0]
    fps  = 30

    print(f"\n{'='*55}\nProcessing: {name}\n{'='*55}")
    print(f"Fusion mode: {fusion_mode}")

    # Extract frames (needed for video-llm mode)
    frame_paths = None
    if fusion_mode == "video-llm":
        print("- Extracting frames for Video-LLM (4 frames at 384px)...")
        frame_dir = os.path.join("temp", "frames", stem)
        frame_data = extract_frames(video_path, frame_dir, every_n=30)
        frame_paths = [f["path"] for f in frame_data]
        print(f"  Extracted {len(frame_paths)} frames")

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
    transitions   = get_scene_transitions(scene_rows)
    object_stats  = get_object_stats(detections)
    person_events = get_event_durations(detections, "person", fps=fps)

    print_temporal_report(transitions, object_stats)

    if person_events:
        print("\n--- Person on screen ---")
        for e in person_events:
            print(f"  {e['start_sec']}s → {e['end_sec']}s ({e['duration_sec']}s)")

    # Transcription
    print("\n- Running Whisper...")
    audio = transcribe_video(video_path)
    print(f"  Transcript: {audio['transcript'] or '[no speech]'}")

    # NLP analysis
    print("- Running NLP analysis...")
    nlp = nlp_results.get(name) if nlp_results else analyze_transcript(audio["transcript"])

    print(f"  Sentiment: {nlp['sentiment']['label']} ({nlp['sentiment']['score']})")
    print(f"  Keywords:  {', '.join(nlp['keywords']) or 'none'}")
    if nlp.get("topic"):
        print(f"  Topic:     {', '.join(nlp['topic']['words'][:4])}")

    # Fusion (dispatches to rule-based or video-llm)
    print(f"\n- Running fusion ({fusion_mode})...")
    summary = fuse(
        video_name     = name,
        detection_rows = detections,
        transcript     = audio["transcript"],
        frame_paths    = frame_paths,
        scene_rows     = scene_rows,
        transitions    = transitions,
        object_stats   = object_stats,
        nlp            = nlp,
        mode           = fusion_mode,
        model_key      = model_key,
    )
    print(f"\n{summary}")

    # Save
    out_file = os.path.join("outputs", f"{stem}_summary.txt")
    with open(out_file, "w") as f:
        f.write(summary)
    print(f"Saved → {out_file}")

    # Note: Video-LLM unloads itself inside generate_llm_summary.
    # No explicit unload needed here.


if __name__ == "__main__":
    ensure_dirs()
    os.makedirs("outputs/segmented_frames", exist_ok=True)

    parser = argparse.ArgumentParser(description="Multimodal Video Analysis Pipeline")
    parser.add_argument("--input", default="data/sample_videos",
                        help="Video file or folder of videos")
    parser.add_argument("--fusion-mode", default="rule-based",
                        choices=["rule-based", "video-llm"],
                        help="Fusion backend: rule-based or video-llm")
    parser.add_argument("--model-key", default="qwen2-vl-2b",
                        choices=["qwen2-vl-2b", "qwen2-vl-7b"],
                        help="Video-LLM model size (only used with --fusion-mode video-llm)")
    args = parser.parse_args()

    # Memory warning for video-llm
    if args.fusion_mode == "video-llm" and args.model_key == "qwen2-vl-7b":
        print("\n⚠️  WARNING: Qwen2-VL-7B requires ~15 GB RAM.")
        print("   On 8 GB systems this WILL fail. Use --model-key qwen2-vl-2b instead.")
        print("   Continuing with 2B model...\n")
        args.model_key = "qwen2-vl-2b"

    # collect all video paths
    if os.path.isfile(args.input):
        video_paths = [args.input]
    else:
        video_paths = list_videos(args.input)

    # corpus-level NLP (train once across all transcripts) 
    print("\n- Collecting transcripts for corpus-level NLP...")
    all_transcripts = {}
    for path in video_paths:
        name  = os.path.basename(path)
        audio = transcribe_video(path)
        all_transcripts[name] = audio["transcript"]

    texts = [t for t in all_transcripts.values() if t.strip()]
    nlp_results = {}

    if len(texts) > 1:
        print(f"  Training Word2Vec on {len(texts)} transcripts...")
        train_word2vec(texts)

        print(f"  Training LDA on {len(texts)} transcripts...")
        train_lda(texts, n_topics=min(5, len(texts)))

        tfidf_kws = tfidf_keywords(texts)
        names     = [n for n, t in all_transcripts.items() if t.strip()]

        for name, kws in zip(names, tfidf_kws):
            transcript      = all_transcripts[name]
            nlp             = analyze_transcript(transcript)
            nlp["keywords"] = kws or nlp["keywords"]   # prefer TF-IDF keywords
            nlp_results[name] = nlp
    else:
        # single video — fallback to per-transcript analysis
        for name, transcript in all_transcripts.items():
            nlp_results[name] = analyze_transcript(transcript)

    # Unload Whisper before video-llm to free ~150 MB
    if args.fusion_mode == "video-llm":
        print("\n- Unloading Whisper to free RAM before Video-LLM...")
        from src.audio.transcribe import unload_model
        unload_model()

    # run full pipeline per video 
    for path in video_paths:
        run(path, nlp_results=nlp_results,
            fusion_mode=args.fusion_mode, model_key=args.model_key)