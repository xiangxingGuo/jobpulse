from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from src.db import get_conn, init_db


@dataclass
class JobFetchResult:
    job_id: str
    source: str
    jd_text: str
    jd_path: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobFetchService:
    """
    Service for fetching job data.
    It returns only the data needed by orchestration layers.
    """

    async def fetch(self, job_id: str, source: str = "handshake") -> JobFetchResult:
        source = (source or "handshake").strip().lower()

        if source == "handshake":
            return await self._fetch_handshake(job_id)
        if source == "greenhouse":
            return await self._fetch_greenhouse(job_id)
        if source == "indeed":
            return await self._fetch_indeed(job_id)

        raise ValueError(f"Unsupported source: {source}")

    async def _fetch_handshake(self, job_id: str) -> JobFetchResult:
        conn = get_conn()
        try:
            init_db()

            row = conn.execute(
                """
                SELECT job_id, description
                FROM jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()

            if not row:
                raise RuntimeError(f"No job found with job_id: {job_id}")

            jd_text = str(row[1] or "")
            md5 = hashlib.md5(jd_text.encode("utf-8", errors="ignore")).hexdigest()

            return JobFetchResult(
                job_id=job_id,
                source="handshake",
                jd_text=jd_text,
                jd_path=None,
                meta={
                    "source": "handshake",
                    "len": len(jd_text),
                    "md5": md5,
                },
            )
        finally:
            conn.close()

    async def _fetch_greenhouse(self, job_id: str) -> JobFetchResult:
        raise NotImplementedError("Greenhouse fetch is not implemented yet.")

    async def _fetch_indeed(self, job_id: str) -> JobFetchResult:
        raise NotImplementedError("Indeed fetch is not implemented yet.")
