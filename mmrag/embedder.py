"""EmbeddingGemma wrappers.

Two interchangeable backends produce L2-normalised vectors in the same space:

* ``GGUFEmbedder``   - quantised EmbeddingGemma GGUF served through llama.cpp
                       (``llama-cpp-python``). Ungated, fast on CPU.
* ``HFEmbedder``     - ``google/embeddinggemma-300m`` through Hugging Face
                       ``transformers`` / ``sentence-transformers``. Gated:
                       requires accepting the licence and an ``HF_TOKEN``.

EmbeddingGemma is trained with task prompts, so queries and documents are
prefixed differently (see the model card).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT = "title: {title} | text: "

GGUF_MODELS: dict[str, tuple[str, str]] = {
    "Q8_0 (ggml-org, 330 MB)": ("ggml-org/embeddinggemma-300M-GGUF", "embeddinggemma-300M-Q8_0.gguf"),
    "Q4_0 (unsloth, 280 MB)": ("unsloth/embeddinggemma-300m-GGUF", "embeddinggemma-300m-Q4_0.gguf"),
    "BF16 (unsloth, 620 MB)": ("unsloth/embeddinggemma-300m-GGUF", "embeddinggemma-300M-BF16.gguf"),
    "F32 (unsloth, 1.2 GB)": ("unsloth/embeddinggemma-300m-GGUF", "embeddinggemma-300M-F32.gguf"),
}

HF_MODEL_ID = "google/embeddinggemma-300m"

# Matryoshka dimensions supported by EmbeddingGemma.
MRL_DIMS = (768, 512, 256, 128)


def _normalise(vectors: np.ndarray, dim: int | None) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if dim is not None and dim < vectors.shape[1]:
        vectors = vectors[:, :dim]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


@dataclass
class BaseEmbedder:
    dim: int = 768
    batch_size: int = 16

    def _encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str], titles: Sequence[str] | None = None) -> np.ndarray:
        titles = titles or ["none"] * len(texts)
        prompts = [DOC_PROMPT.format(title=t or "none") + x for t, x in zip(titles, texts)]
        out = [self._encode(prompts[i : i + self.batch_size]) for i in range(0, len(prompts), self.batch_size)]
        return _normalise(np.vstack(out), self.dim) if out else np.zeros((0, self.dim), np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return _normalise(self._encode([QUERY_PROMPT + query]), self.dim)[0]


class GGUFEmbedder(BaseEmbedder):
    def __init__(self, variant: str = next(iter(GGUF_MODELS)), dim: int = 768, n_ctx: int = 2048, n_threads: int | None = None):
        super().__init__(dim=dim)
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        repo_id, filename = GGUF_MODELS[variant]
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.llm = Llama(
            model_path=path,
            embedding=True,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            n_ubatch=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self.name = f"{repo_id}/{filename}"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        # One call per text keeps llama.cpp's sequence handling simple and
        # avoids exceeding n_ctx when batching long chunks.
        return np.array([self.llm.embed(t, truncate=True) for t in texts], dtype=np.float32)


class HFEmbedder(BaseEmbedder):
    def __init__(self, model_id: str = HF_MODEL_ID, dim: int = 768, device: str | None = None):
        super().__init__(dim=dim)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self.name = model_id

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=False)
