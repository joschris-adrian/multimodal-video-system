import argparse
import os
from src.utils.file_utils import list_videos, ensure_dirs
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
from src.fusion.summarize import generate_summary


def run(video_path, nlp_results=None):
    name = os.path.basename(video_path)
    stem = os.path.splitext(name)[0]
    fps  = 30

    print(f"\n{'='*55}\nProcessing: {name}\n{'='*55}")

    # Object detection
    print("▶ Running YOLOv8 detection...")
    detections = detect_objects(video_path, every_n=30)

    # Scene classification
    print("▶ Running scene classification...")
    scene_rows = classify_video(video_path, every_n=30)

    # Segmentation
    print("▶ Running segmentation...")
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
    print("\n▶ Running Whisper...")
    audio = transcribe_video(video_path)
    print(f"  Transcript: {audio['transcript'] or '[no speech]'}")

    # NLP analysis
    print("▶ Running NLP analysis...")
    nlp = nlp_results.get(name) if nlp_results else analyze_transcript(audio["transcript"])

    print(f"  Sentiment: {nlp['sentiment']['label']} ({nlp['sentiment']['score']})")
    print(f"  Keywords:  {', '.join(nlp['keywords']) or 'none'}")
    if nlp.get("topic"):
        print(f"  Topic:     {', '.join(nlp['topic']['words'][:4])}")

    # Fusion
    summary = generate_summary(
        video_name     = name,
        detection_rows = detections,
        transcript     = audio["transcript"],
        scene_rows     = scene_rows,
        transitions    = transitions,
        object_stats   = object_stats,
        nlp            = nlp,
    )
    print(f"\n{summary}")

    # Save
    out_file = os.path.join("outputs", f"{stem}_summary.txt")
    with open(out_file, "w") as f:
        f.write(summary)
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    ensure_dirs()
    os.makedirs("outputs/segmented_frames", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_videos")
    args = parser.parse_args()

    # collect all video paths
    if os.path.isfile(args.input):
        video_paths = [args.input]
    else:
        video_paths = list_videos(args.input)

    # corpus-level NLP (train once across all transcripts) 
    print("\n▶ Collecting transcripts for corpus-level NLP...")
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

    # run full pipeline per video 
    for path in video_paths:
        run(path, nlp_results=nlp_results)