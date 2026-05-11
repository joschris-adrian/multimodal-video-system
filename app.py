import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image

# page config 
st.set_page_config(
    page_title = "Multimodal Video Analysis",
    page_icon  = "🎬",
    layout     = "wide",
)

st.title("🎬 Multimodal Video Analysis System")
st.caption("YOLOv8 · EfficientNet · Whisper · Stable Diffusion · BERT · LDA")

# sidebar 
with st.sidebar:
    st.header("⚙️ Settings")
    every_n      = st.slider("Sample every N frames", 10, 60, 30)
    conf         = st.slider("Detection confidence",  0.1, 0.9, 0.25)
    run_tracking = st.checkbox("Enable object tracking",          value=False)
    run_gen      = st.checkbox("Generate image from summary",     value=False)
    use_bert     = st.checkbox("Use BERT embeddings (slow on CPU)", value=False)
    sd_steps     = st.slider("Image generation steps", 2, 8, 4,
                             disabled=not run_gen)
    st.divider()
    st.caption("Models download automatically on first run.")
    st.caption("BERT: ~80MB · Stable Diffusion: ~3GB")

# upload 
uploaded = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded is None:
    st.info("👆 Upload a video to get started.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    tmp.write(uploaded.read())
    video_path = tmp.name

st.video(uploaded)
st.divider()

# run pipeline 
if st.button("▶ Run Analysis", type="primary", use_container_width=True):

    from src.vision.detect_objects  import detect_objects
    from src.vision.classify_scene  import classify_video
    from src.vision.segment_objects import segment_video
    from src.audio.transcribe       import transcribe_video
    from src.audio.nlp_analysis     import analyze_transcript
    from src.temporal.aggregator    import (
        get_scene_transitions, get_object_stats
    )
    from src.fusion.summarize import generate_summary

    os.makedirs("outputs", exist_ok=True)

    # 1. Detection
    with st.spinner("Running YOLOv8 detection..."):
        detections = detect_objects(video_path, conf=conf, every_n=every_n)
    st.success(f"✅ Detection — {len(detections)} frames sampled")

    # 2. Scene classification
    with st.spinner("Classifying scenes..."):
        scene_rows = classify_video(video_path, every_n=every_n)
    st.success("✅ Scene classification complete")

    # 3. Segmentation
    seg_dir = "outputs/streamlit_seg"
    with st.spinner("Running segmentation..."):
        segment_video(video_path, seg_dir, every_n=every_n)
    st.success("✅ Segmentation complete")

    # 4. Transcription
    with st.spinner("Transcribing audio with Whisper..."):
        audio = transcribe_video(video_path)
    st.success("✅ Transcription complete")

    # 5. NLP
    with st.spinner("Running NLP analysis..."):
        nlp = analyze_transcript(audio["transcript"], use_bert=use_bert)
    st.success("✅ NLP analysis complete")

    # 6. Temporal
    transitions  = get_scene_transitions(scene_rows)
    object_stats = get_object_stats(detections)

    # 7. Fusion
    summary = generate_summary(
        video_name     = uploaded.name,
        detection_rows = detections,
        transcript     = audio["transcript"],
        scene_rows     = scene_rows,
        transitions    = transitions,
        object_stats   = object_stats,
        nlp            = nlp,
    )

    # ── display results ───────────────────────────────────────────────────────
    st.divider()
    st.header("📋 Results")

    col1, col2 = st.columns(2)

    with col1:
        # --- summary ---
        st.subheader("🧠 Summary")
        st.code(summary, language="text")

        # --- transcript ---
        st.subheader("🎙️ Transcript")
        if audio["transcript"]:
            st.write(audio["transcript"])
            badge = {"good": "🟢", "noisy": "🟡", "too_short": "🟠", "empty": "🔴"}
            st.caption(
                f"Quality: {badge.get(audio['quality'], '⚪')} {audio['quality']}  "
                f"| Words: {audio['word_count']}"
            )
        else:
            st.warning("No speech detected")

        # --- NLP ---
        st.subheader("🔍 NLP Analysis")

        m1, m2, m3 = st.columns(3)
        m1.metric("Sentiment",  nlp["sentiment"]["label"].capitalize())
        m2.metric("Words",      nlp["word_count"])
        m3.metric("Sentences",  nlp["sentence_count"])

        if nlp["keywords"]:
            st.write("**Keywords:**", "  ·  ".join(nlp["keywords"]))

        if nlp.get("topic") and nlp["topic"].get("words"):
            st.write(
                "**Dominant Topic:**",
                "  ·  ".join(nlp["topic"]["words"][:5]),
                f"_(weight: {nlp['topic']['weight']})_"
            )

        if use_bert and nlp.get("bert_embedding") is not None:
            emb  = nlp["bert_embedding"]
            norm = float(np.linalg.norm(emb))
            st.caption(f"BERT embedding — dim: {emb.shape[0]} | norm: {norm:.2f}")

        # --- object persistence ---
        st.subheader("📊 Object Persistence")
        if object_stats.get("per_object"):
            for label, stats in sorted(
                object_stats["per_object"].items(),
                key=lambda x: -x[1]["presence_pct"]
            ):
                st.progress(
                    stats["presence_pct"] / 100,
                    text=f"{label} — {stats['presence_pct']}%"
                )

    with col2:
        # --- annotated frames ---
        st.subheader("🖼️ Annotated Frames")
        ann_dir = os.path.join(
            "outputs", "annotated_frames",
            os.path.splitext(os.path.basename(video_path))[0]
        )
        if os.path.isdir(ann_dir):
            frame_files = sorted(os.listdir(ann_dir))[:6]
            cols = st.columns(2)
            for i, f in enumerate(frame_files):
                img = Image.open(os.path.join(ann_dir, f))
                cols[i % 2].image(img, width=300)

        # --- segmented frames ---
        st.subheader("🎭 Segmented Frames")
        if os.path.isdir(seg_dir):
            seg_files = sorted(os.listdir(seg_dir))[:4]
            cols = st.columns(2)
            for i, f in enumerate(seg_files):
                img = Image.open(os.path.join(seg_dir, f))
                cols[i % 2].image(img, width=300)

    # --- scene timeline ---
    st.divider()
    st.subheader("⏱️ Scene Timeline")
    if transitions:
        for t in transitions:
            st.write(
                f"**Frames {t['start_frame']}–{t['end_frame']}** "
                f"→ {t['scene']}"
            )
    else:
        st.write("No scene transitions detected.")

    # --- tracking ---
    if run_tracking:
        st.divider()
        st.subheader("🎯 Object Tracking")
        from src.tracking.track_objects import track_video
        track_out = "outputs/tracked.mp4"
        with st.spinner("Running tracker..."):
            track_video(video_path, track_out, conf=conf)
        if os.path.exists(track_out):
            st.video(track_out)

    # --- image generation ---
    if run_gen:
        st.divider()
        st.subheader("🎨 Generated Image")
        st.warning(
            "⚠️ Image generation runs on CPU and takes ~2–4 min. "
            "Enable only if you want to wait."
        )
        from src.generation.generate_image import build_prompt, generate_image
        from collections import Counter

        all_labels     = [l for r in detections for l in r["labels"]]
        top_objects    = [o for o, _ in Counter(all_labels).most_common(3)]
        dominant_scene = (
            Counter(r["scene"] for r in scene_rows).most_common(1)[0][0]
            if scene_rows else "unknown"
        )

        prompt = build_prompt(top_objects, dominant_scene, audio["transcript"])
        st.caption(f"Prompt: _{prompt}_")

        with st.spinner("Generating image with Stable Diffusion..."):
            img = generate_image(prompt, steps=sd_steps)
        st.image(img, width=600)

    # cleanup
    os.unlink(video_path)
    st.balloons()