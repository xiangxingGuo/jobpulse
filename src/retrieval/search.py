from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.db import fetch_jobs_for_retrieval
from src.retrieval.embed import EmbeddingModel
from src.retrieval.faiss_index import JobFaissIndex

DEFAULT_INDEX_DIR = Path("data/vectors")


class JobSearchService:
    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.index_dir = Path(index_dir)
        self.model = EmbeddingModel(model_name)
        self.index = JobFaissIndex.load(self.index_dir)

        # build quick lookup by job_id from metadata
        self.meta_by_job_id: dict[str, dict[str, Any]] = {
            str(m["job_id"]): m for m in self.index.meta if m.get("job_id")
        }
        self.rowid_by_job_id: dict[str, int] = {
            str(m["job_id"]): i for i, m in enumerate(self.index.meta) if m.get("job_id")
        }

    @lru_cache(maxsize=128)
    def _encode_query_cached(self, query: str) -> tuple[float, ...]:
        vec = self.model.encode([query])
        return tuple(vec[0])

    def _encode_query(self, query: str) -> np.ndarray:
        arr = np.array([self._encode_query_cached(query)], dtype="float32")
        return arr

    def search_jobs(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        qvec = self._encode_query(query)
        results = self.index.search(qvec, top_k=top_k)
        return results

    def similar_jobs(self, job_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        job_id = str(job_id)

        if job_id not in self.rowid_by_job_id:
            raise ValueError(f"job_id not found in index: {job_id}")

        row_id = self.rowid_by_job_id[job_id]

        # reconstruct only works for flat indexes like IndexFlatIP
        base_vec = self.index.index.reconstruct(row_id).astype("float32")[None, :]
        results = self.index.search(base_vec, top_k=top_k + 1)

        filtered = [r for r in results if str(r.get("job_id")) != job_id]
        return filtered[:top_k]


def get_job_by_id(job_id: str) -> dict[str, Any] | None:
    rows = fetch_jobs_for_retrieval(limit=None)
    for row in rows:
        if str(row.get("job_id")) == str(job_id):
            return row
    return None
