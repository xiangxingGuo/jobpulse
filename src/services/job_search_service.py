from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retrieval.search import (
    JobSearchService as RetrievalJobSearchService,
    get_job_by_id as retrieval_get_job_by_id,
)


DEFAULT_INDEX_DIR = Path("data/vectors")


@dataclass
class JobSearchHit:
    job_id: str
    score: float
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    content_hash: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobSearchService:
    """
    Service-layer facade over src.retrieval.search.

    Goal:
    - keep retrieval implementation in src.retrieval
    - expose stable service methods for API / graph / future skill-gap flows
    """

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self._retrieval = RetrievalJobSearchService(
            index_dir=self.index_dir,
            model_name=self.model_name,
        )

    def search_jobs(self, query: str, top_k: int = 10) -> List[JobSearchHit]:
        query = (query or "").strip()
        if not query:
            return []

        rows = self._retrieval.search_jobs(query=query, top_k=top_k)
        return [self._normalize_hit(row) for row in rows]

    def similar_jobs_for_job(self, job_id: str, top_k: int = 5) -> List[JobSearchHit]:
        job_id = str(job_id).strip()
        if not job_id:
            raise ValueError("job_id is empty")

        rows = self._retrieval.similar_jobs(job_id=job_id, top_k=top_k)
        return [self._normalize_hit(row) for row in rows]

    def get_job_by_id(self, job_id: str) -> Dict[str, Any] | None:
        job_id = str(job_id).strip()
        if not job_id:
            return None
        return retrieval_get_job_by_id(job_id)

    def get_market_context_summary(
        self,
        *,
        job_id: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Lightweight market-context summary for downstream skill-gap analysis.

        v1:
        - retrieves similar jobs
        - aggregates visible metadata only
        - no LLM required
        """
        similar = self.similar_jobs_for_job(job_id=job_id, top_k=top_k)

        companies = []
        titles = []
        locations = []

        for hit in similar:
            if hit.company:
                companies.append(hit.company)
            if hit.title:
                titles.append(hit.title)
            if hit.location:
                locations.append(hit.location)

        return {
            "anchor_job_id": str(job_id),
            "top_k": int(top_k),
            "similar_jobs": [hit.to_dict() for hit in similar],
            "companies": _unique_keep_order(companies),
            "titles": _unique_keep_order(titles),
            "locations": _unique_keep_order(locations),
        }

    def _normalize_hit(self, row: Dict[str, Any]) -> JobSearchHit:
        return JobSearchHit(
            job_id=str(row.get("job_id", "")),
            score=float(row.get("score", 0.0)),
            title=_clean_optional_str(row.get("title")),
            company=_clean_optional_str(row.get("company")),
            location=_clean_optional_str(row.get("location")),
            url=_clean_optional_str(row.get("url")),
            content_hash=_clean_optional_str(row.get("content_hash")),
            raw=row,
        )


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out