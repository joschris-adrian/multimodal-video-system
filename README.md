# Multimodal Video Analysis System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple)
![Whisper](https://img.shields.io/badge/Whisper-openai-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

A modular Python pipeline that runs object detection, scene classification,
segmentation, speech transcription, object tracking, and image generation on
video files — fusing everything into structured per-clip summaries.

---

## Stack

| Component            | Library                              |
|----------------------|--------------------------------------|
| Object detection     | YOLOv8 (ultralytics)                 |
| Segmentation         | YOLOv8-seg (ultralytics)             |
| Scene classification | EfficientNet-B0 (torchvision)        |
| Transcription        | Whisper (openai-whisper)             |
| NLP                  | scikit-learn · gensim · transformers |
| Object tracking      | IoU tracker (custom)                 |
| Image generation     | Stable Diffusion (diffusers)         |
| Video-LLM Fusion     | Qwen2-VL (transformers)              |
| UI                   | Streamlit                            |
| Frame / audio I/O    | OpenCV + moviepy + scipy             |
| Data                 | pandas                               |
	

---

## Setup

```bash
git clone https://github.com/joschris-adrian/multimodal-video-system.git
cd multimodal-video-system
pip install -r requirements.txt
```

---

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload any video and hit **Run Analysis**.

### CLI

```bash
# single video (rule-based — default)
python run_pipeline.py --input path/to/clip.mp4

# single video (video-llm — requires ~5GB free RAM)
python run_pipeline.py --input path/to/clip.mp4 --fusion-mode video-llm

# whole folder
python run_pipeline.py --input data/sample_videos
```

---

## UI

Upload any video and hit **Run Analysis**. The app runs the full pipeline
and displays all results inline.

| Section             | Content                                       |
|---------------------|-----------------------------------------------|
| Summary             | One-liner + full fusion output                |
| Transcript          | Whisper text + quality badge                  |
| Object persistence  | Progress bars per detected object             |
| Annotated frames    | YOLOv8 bounding boxes, 6 sampled frames       |
| Segmented frames    | Mask overlays for person/car, 4 frames        |
| Scene timeline      | Frame ranges + scene labels                   |
| Tracking video      | Playable video with persistent track IDs      |
| Generated image     | Stable Diffusion output from pipeline summary |

---

## Output

```
outputs/
├── clip_summary.txt              ← structured fusion summary
├── whisper_report.csv            ← transcription quality for all clips
├── annotated_frames/clip/        ← YOLOv8 bounding boxes per frame
├── segmented_frames/clip/        ← YOLOv8-seg masks per frame
├── tracked.mp4                   ← video with persistent track IDs
└── generated.png                 ← Stable Diffusion image
```

Example `clip_summary.txt`:

```
FILE: v_Surfing_g22_c01.avi
Summary: A person is surfing in a sports venue.
Detected: person, surfboard
Scene: sports
Audio: The idea was to get over there.
Empty frames: 0.0%
Scene timeline:
  0–60: sports
  60–90: beach
Always present: person
```

---

## How the fusion layer works

```
The pipeline offers two fusion backends, selectable via CLI flag or Streamlit sidebar.

### 1. Rule-Based (Default)

Fast, template-based reasoning with zero extra memory:

```
subject     ← is there a person in the detections?
environment ← what does the scene classifier say?
action      ← what keywords appear in the transcript?

→ "A person is walking in an urban street."
```

Scene is inferred from EfficientNet when confident, with an object-based fallback when the classifier returns `unknown`.

### 2. Video-LLM (Qwen2-VL)

Feeds sampled frames + transcript + detections directly to a vision-language model for deeply contextual summaries:

```
frames  ← 4 sampled frames at 384px
context ← transcript + detections + scene + NLP keywords
model   ← Qwen2-VL-2B-Instruct (fp16)

→ "A person in a wetsuit rides a breaking wave toward a sandy beach."
```

Memory-optimised for 8 GB RAM: automatically unloads Whisper before loading the VLM, uses `low_cpu_mem_usage`, and unloads the VLM immediately after inference. Falls back to rule-based if RAM is insufficient.
```
## Image Generation

The system builds a Stable Diffusion prompt automatically from
pipeline outputs — no manual prompt writing needed.

```
subject     ← "A person" if person detected, else "A scene"
action      ← matched from transcript keywords, then from detected objects
environment ← from scene classifier, with object-based fallback
```

Examples:

```
objects=["person", "hair drier"]   → "A person getting a blowdry in a modern indoor room"
objects=["person", "surfboard"]    → "A person surfing in a sports venue"
transcript="She was walking fast"  → "A person walking in an urban street"
```

Runs locally via `stabilityai/sd-turbo` (4 steps, CPU-friendly).
Alternatively swap to the Hugging Face Inference API for instant
results without local compute — get a free token at
`huggingface.co/settings/tokens`.

---

## NLP Layer

The pipeline runs a full NLP analysis on every Whisper transcript,
combining classical and deep learning approaches.

| Technique     | Library      | Output                              |
|---------------|--------------|-------------------------------------|
| TF-IDF        | scikit-learn | keyword ranking per clip            |
| Word2Vec      | gensim       | 64-dim mean-pool embedding per clip |
| BERT (MiniLM) | transformers | 384-dim sentence embedding          |
| LDA           | scikit-learn | topic distribution per clip         |
| Sentiment     | rule-based   | positive / neutral / negative       |

TF-IDF and LDA are trained across the full video corpus in one pass,
so keywords and topics are relative to the entire dataset — not just
a single clip. Word2Vec and BERT embeddings enable clip similarity
search and semantic comparison between videos.

Example NLP output per clip:

```
Keywords:  walking, street, city, busy, traffic
Sentiment: positive 😊
Topic:     street, car, road, urban, person
```

BERT embeddings are optional (disabled by default in the UI) since
they require ~80MB and add latency on CPU.

## Object Tracking

A lightweight IoU-based tracker assigns persistent IDs to objects
across frames — no additional dependencies needed.

```
person #0 ── tracked across 42 frames
person #1 ── appeared at frame 90, tracked for 18 frames
car    #2 ── tracked across 60 frames
```

Enable in the Streamlit sidebar or call directly:

```python
from src.tracking.track_objects import track_video
track_video("clip.mp4", "outputs/tracked.mp4")
```

---

## Temporal EDA

The aggregator tracks what happens over time across sampled frames:

```
--- Scene Timeline ---
  Frame     0–60  │ sports
  Frame    60–90  │ beach

--- Object Persistence ---
  person               100.0%  ████████████████████
  surfboard             66.7%  █████████████

Always present:  person
Briefly seen:    bird
```

---

## Project Structure

```
src/
├── video/
│   └── extract_frames.py         ← OpenCV frame extraction
├── vision/
│   ├── detect_objects.py         ← YOLOv8 detection + annotated frames
│   ├── classify_scene.py         ← EfficientNet scene classification
│   └── segment_objects.py        ← YOLOv8-seg masks for key objects
├── temporal/
│   └── aggregator.py             ← scene transitions, object persistence,
│                                    event durations
├── audio/
│   ├── transcribe.py             ← Whisper via moviepy (no ffmpeg)
│   └── nlp_analysis.py           ← TF-IDF, Word2Vec, BERT, LDA, sentiment
├── fusion/
│   └── summarize.py              ← structured reasoning → one-liner summary
├── generation/
│   └── generate_image.py         ← Stable Diffusion prompt builder + inference
├── tracking/
│   └── track_objects.py          ← IoU tracker with persistent IDs
└── utils/
    └── file_utils.py             ← shared helpers
app.py                            ← Streamlit UI
run_pipeline.py                   ← CLI entrypoint
```

---

## Tests

```bash
pip install pytest
pytest tests/ -v

# run specific sections
pytest tests/ -v -k "temporal"        # aggregator tests
pytest tests/ -v -k "scene"           # scene classification tests
pytest tests/ -v -k "prompt"          # image generation prompt tests
pytest tests/ -v -k "tracker"         # tracking tests
pytest tests/ -v -k "sentiment"       # NLP sentiment tests
pytest tests/ -v -k "bert"            # BERT embedding tests
pytest tests/ -v -k "lda"             # topic modelling tests
pytest tests/ -v -k "not bert"        # skip slow BERT tests
pytest tests/ -v -k "not video"       # skip slow video tests
```

The test suite uses a synthetic video generated by OpenCV - no real video files needed.

---

## Models

All models download automatically on first run.

| Model            | Size   | Purpose                  |
|------------------|--------|--------------------------|
| `yolov8n.pt`     | 6 MB   | Object detection         |
| `yolov8n-seg.pt` | 7 MB   | Segmentation             |
| EfficientNet-B0  | 21 MB  | Scene classification     |
| Whisper `base`   | 74 MB  | Transcription            |
| MiniLM-L6-v2     | 80 MB  | BERT sentence embeddings |
| Qwen2-VL-2B-Instruct|	~3.5GB|	Video-LLM fusion       |
| SD-Turbo         | ~3 GB  | Image generation         |

Swap detection/segmentation to `yolov8s` or `yolov8m` for better
accuracy. Swap Whisper to `small` or `medium` for better transcription.

---

## Notes

- `every_n=30` samples 1 frame per second at 30fps — adjust in the
  sidebar to trade speed vs. coverage
- Image generation is disabled by default in the UI — enable in the
  sidebar when needed (takes ~2–4 min on CPU)
- `data/` and `outputs/` are excluded from git — videos are large,
  outputs are generated
- Models are excluded from git — they download on demand
- Video-LLM mode requires ~5 GB free RAM. On 8 GB systems, Whisper is 
automatically unloaded before loading the VLM, and the 7B model variant 
is hidden from the UI to prevent out-of-memory crashes.
---

## Next Steps

### Quick Hits

- Add `__init__.py` to `src/learning/` and `src/audio/nlu_analysis.py`
  and wire NLU toggle into the Streamlit sidebar — one checkbox, three
  model downloads, zero pipeline changes required
- Make `every_n` and `conf` configurable via environment variables so
  the CLI and UI share the same defaults without editing code
- Add a `GET /health` check to confirm all models are loaded before
  the Streamlit UI allows analysis to run
- Fix `use_column_width` deprecation warnings across `app.py` — replace
  all occurrences with `width=700`
- Add a `POST /reload-config` pattern so sidebar settings take effect
  without restarting the Streamlit server
- Make Whisper model size configurable via environment variable —
  replace hardcoded `"base"` with `os.environ.get("WHISPER_MODEL", "base")`
- Add a ping or health check per model on startup and log a clear
  warning per unavailable model rather than failing silently at runtime
- Make the relevance score threshold in the NLP layer configurable
  via environment variable
- Add a `Dockerfile` and `docker-compose.yml` (CPU and GPU variants)
  to freeze dependencies and eliminate local environment issues
- Add a GitHub Actions workflow to run the 90+ pytest suite and lint
  checks on every push to prevent regressions

---

### Bigger Changes

- **NLU layer** — extend NLP with named entity recognition
  (`dslim/bert-base-NER`), fine-grained emotion detection
  (`emotion-english-distilroberta-base`), and zero-shot intent
  classification (`facebook/bart-large-mnli`), feeding entities,
  emotion, and intent into the fusion summary
- **Zero-shot retrieval** — integrate pre-trained contrastive models
  (e.g. CLIP, InternVideo2) to enable text-to-clip search from a text
  query against the existing EfficientNet and MiniLM embeddings
  without training from scratch
- **Action recognition** — replace object-based action inference with
  off-the-shelf video transformers (e.g. HuggingFace VideoMAE
  fine-tuned on Kinetics) for clip-level action classification,
  removing dependency on transcript keywords
- **Vector database** — store BERT and contrastive embeddings in
  ChromaDB or FAISS for fast nearest-neighbour retrieval across large
  video corpora, replacing the in-memory cosine search
- **RAG over transcripts** — index all transcripts in a vector store
  and answer natural language queries across the full corpus,
  e.g. "which clips show someone cooking indoors?"
- **Fine-tuned summariser** — replace rule-based fusion with a LoRA
  fine-tuned language model trained on (pipeline output → natural
  language summary) pairs, using LLM-as-judge scoring to benchmark
  against the rule-based baseline
- **Pipeline orchestration** — replace sequential execution and
  hardcoded `asyncio` with a DAG executor (e.g. LangGraph, Prefect)
  to natively run Vision and Audio branches in parallel, handle
  conditional routing, and cache model outputs
- **Advanced chunking** — replace fixed frame sampling with
  scene-boundary detection so frames are sampled at semantically
  meaningful transitions rather than every N frames
- **Monitoring and observability** — add structured JSON logging per
  module with timestamps, inference latency, and detection counts
  exportable to a dashboard
- **UI migration** — migrate from Streamlit to a decoupled FastAPI
  backend and Next.js (or robust Gradio) frontend to better handle
  complex state, heavy media playback, and concurrent users