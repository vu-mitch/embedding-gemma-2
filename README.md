# embedding-gemma-2

A Streamlit demo of a **multimodal retrieval pipeline** built on
[EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m). Upload up to
3 PDFs, 3 audio files and 3 videos, then retrieve the most relevant passages,
audio segments and video moments for a prompt.

Semantic similarity is the first pass. The second pass is explicit temporal
reasoning over visual, acoustic and spoken events: a vibration that begins
before an operator report, an unusual sound that sits inside that vibration,
and a 15 second window a merely similar clip can still fail.

## How it works

EmbeddingGemma is a text embedding model, so each modality is first projected
into text and then embedded into a **single shared vector space**:

| Modality | Extraction (Hugging Face `transformers` and tools) | Chunk locator |
|----------|---------------------------------------|---------------|
| PDF      | `pypdf` page text, overlapping word chunks | page number |
| Audio    | Whisper ASR (`transformers` pipeline) with timestamps, grouped into windows | start-end time |
| Video    | Whisper on the audio track + BLIP captions of sampled keyframes | start-end time |

Documents are embedded with the `title: <file> | text: ...` prompt and queries
with `task: search result | query: ...`, as recommended on the model card.
Vectors are L2-normalised and searched with cosine similarity. Results show the
best-matching files, then the top passages with an audio/video player that starts
at the matching timestamp.

### Embedding backends

* **GGUF (llama.cpp)**, the default: quantised EmbeddingGemma from
  `ggml-org/embeddinggemma-300M-GGUF` (Q8_0) or `unsloth/embeddinggemma-300m-GGUF`
  (Q4_0 / BF16 / F32), loaded with `llama-cpp-python`. These repos are not gated.
* **Transformers (HF)**: `google/embeddinggemma-300m` through
  `sentence-transformers`/`transformers`. This model is gated, so accept the licence on
  Hugging Face and export `HF_TOKEN` first.

You can also choose a Matryoshka dimension (768 / 512 / 256 / 128) to trade accuracy for speed.
A **Lexical (offline)** backend is a hashed bag-of-words with the same prompts,
used by the manufacturing example so the temporal layer can be tried without a model download.

## Temporal reasoning

Enterprise multimodal search already retrieves the clip that *mentions* a vibration.
The testable difference here is a constraint over the timeline:

```
t_vibration < t_report
vibration within 15 seconds before the report
acoustic anomaly during the vibration
```

On the figure's recording the vibration starts at 00:10 and the operator report at
00:40, so the order holds and the 15 second window does not. An unusual sound from
00:15–00:30 falls inside the vibration. Two other lines meet the window, including a
2.5 second vibration inside a 5 minute take.

Short events are localized from the recording itself:

| Channel | Detector | What it is for |
|---------|----------|----------------|
| Visual | frame-difference against the clip's own quiet baseline | abnormal vibration, normal operation |
| Acoustic | short-window RMS against that same baseline | unusual sound |
| Spoken | Whisper segments plus domain cues (and, when EmbeddingGemma is loaded, prototype similarity) | operator report |

Motion and loudness keep the tight span. A keyframe caption can confirm the label
without stretching the event out to the keyframe window. Each hit lists the formula,
the timestamps, the pass/fail delta and the evidence text.

### Evaluation

`python -m mmrag.benchmark` prints four metrics against a semantic ablation
(coarse windows ranked by lexical overlap, no constraint check):

* **Temporal IoU** — overlap of the predicted span with the true event
* **Recall@5** — fraction of true vibration spans matched in the top 5 at IoU ≥ 0.5
* **Event-ordering accuracy** — pairwise order of vibration, sound and report
* **Answer correctness** — token F1 of the yes/no answer, including the line that misses the 15 second window

```bash
python -m pytest tests -q
```

In the app, **Load manufacturing example** draws the three timelines and applies the
full constraint. **Run temporal evaluation** shows the same four numbers.
`python scripts/make_manufacturing.py` writes a 50 second clip with the figure's
motion and tone (the spoken report still comes from the transcript).

## Setup

Requires Python 3.10+ and `ffmpeg` on `PATH`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -r requirements.txt
streamlit run app.py
```

Models download from the Hugging Face Hub the first time you use them.

## Try it with generated samples

```bash
sudo apt install espeak-ng          # text-to-speech for the sample audio
python scripts/make_samples.py      # writes 3 PDFs, 3 WAVs, 3 MP4s into ./samples
```

Example prompts: *"how do plants make oxygen"*, *"what happens near an event
horizon"*, *"how to fix a sour espresso shot"*, *"lava coming out of a mountain"*.

## Layout

```
app.py                    Streamlit UI
mmrag/embedder.py         EmbeddingGemma backends, plus an offline lexical embedder
mmrag/extractors.py       PDF / audio / video -> text chunks
mmrag/events.py           manufacturing event catalog and evidence spans
mmrag/signals.py          short-event localization from motion and loudness
mmrag/temporal.py         constraint parser and solver
mmrag/metrics.py          Temporal IoU, Recall@k, ordering, answer F1
mmrag/benchmark.py        figure timeline and the semantic ablation
mmrag/analyse.py          chunks + events for one uploaded file
mmrag/timeline.py         HTML timeline
mmrag/index.py            in-memory cosine-similarity index
scripts/make_samples.py   demo data generator
scripts/make_manufacturing.py  50s figure clip (motion + tone)
```
