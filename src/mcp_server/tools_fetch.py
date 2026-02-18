from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from src.orch.schema import FetchJDOutput


JD_DIR = Path("data/raw/jd_txt")


def fetch_jd(job_id: str, source: Optional[str] = None) -> FetchJDOutput:
    """
    Fetch a cleaned Job Description text for a job_id.

    Args:
      job_id: Job ID string.
      source: Optional source name (handshake/greenhouse/etc).

    Returns:
      {job_id, jd_text, jd_path, meta}
    """
    path = JD_DIR / f"{job_id}.txt"
    jd_text = path.read_text(encoding="utf-8", errors="ignore")

    md5 = hashlib.md5(jd_text.encode("utf-8", errors="ignore")).hexdigest()

    return {
        "job_id": job_id,
        "jd_text": jd_text,
        "jd_path": str(path),
        "meta": {
            "source": source,
            "len": len(jd_text),
            "md5": md5,
        },
    }
