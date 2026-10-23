# embedding-gemma-2

A Streamlit demo of a **multimodal retrieval pipeline** built on
[EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m). Upload up to
3 PDFs, 3 audio files and 3 videos, then retrieve the most relevant passages,
audio segments and video moments for a prompt.

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

## Gleisbau demo

`demo/gleisbau/` is a fictional maintenance corpus for four cases (manual plus
service video, spare-part drawing, archived report plus recorded conversation,
training sheet plus training video). In the app, click **Load Gleisbau demo**,
choose a case, then **Show the Evidence** to open the retrieved page or timestamp.
Regenerate the files with `python scripts/make_gleisbau_demo.py` (`ffmpeg`,
`fpdf2`, `edge-tts`). The papers are labelled as demo material and do not approve
real repairs.

## Layout

```
app.py               Streamlit UI
mmrag/embedder.py    EmbeddingGemma backends (GGUF via llama.cpp, HF via transformers)
mmrag/extractors.py  PDF / audio / video -> text chunks
mmrag/index.py       in-memory cosine-similarity index
scripts/make_samples.py  demo data generator
```
