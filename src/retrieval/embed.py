from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers.utils import logging

logging.set_verbosity_error()  # suppress warnings from transformers about UNEXPECTED_KEYWORD_ARGUMENTS in sentence_transformers


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]

        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(vecs, dtype="float32")
