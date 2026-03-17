from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class JobFaissIndex:
    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.meta: list[dict[str, Any]] = []

    def add(self, vectors: np.ndarray, meta: list[dict[str, Any]]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        self.index.add(vectors)
        self.meta.extend(meta)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")

        scores, ids = self.index.search(query_vec, top_k)
        results: list[dict[str, Any]] = []

        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            if idx >= len(self.meta):
                continue
            results.append(
                {
                    "score": float(score),
                    **self.meta[idx],
                }
            )
        return results

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(out_dir / "jobs.faiss"))

        with (out_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
            for row in self.meta:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, out_dir: str | Path) -> "JobFaissIndex":
        out_dir = Path(out_dir)

        index = faiss.read_index(str(out_dir / "jobs.faiss"))
        obj = cls(index.d)
        obj.index = index

        with (out_dir / "meta.jsonl").open("r", encoding="utf-8") as f:
            obj.meta = [json.loads(line) for line in f if line.strip()]

        index_size = obj.index.ntotal
        meta_size = len(obj.meta)
        if index_size != meta_size:
            raise ValueError(
                f"FAISS index/meta mismatch: index.ntotal={index_size}, meta_rows={meta_size}. "
                "Rebuild the vector index using scripts/build_vector_index.py"
            )

        return obj