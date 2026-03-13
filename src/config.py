# src/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import os
import uuid

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

@dataclass(frozen=True)
class ScrapeConfig:
    # Identity
    run_id: str = field(default_factory=lambda: os.getenv("RUN_ID", uuid.uuid4().hex[:10]))

    # Browser
    headless: bool = _env_bool("HEADLESS", False)

    # Scope
    pages: int = _env_int("PAGES", 1)
    per_page: int = _env_int("PER_PAGE", 25)
    limit: int = _env_int("LIMIT", 100)

    # Playwright timeouts (ms)
    goto_timeout_ms: int = _env_int("GOTO_TIMEOUT_MS", 20000)
    networkidle_timeout_ms: int = _env_int("NETWORKIDLE_TIMEOUT_MS", 15000)
    selector_timeout_ms: int = _env_int("SELECTOR_TIMEOUT_MS", 8000)

    # Rate limiting / politeness
    sleep_range_sec: Tuple[float, float] = (_env_float("SLEEP_MIN", 1.5), _env_float("SLEEP_MAX", 3.8))
    extra_pause_prob: float = _env_float("EXTRA_PAUSE_PROB", 0.12)
    extra_pause_range_sec: Tuple[float, float] = (_env_float("EXTRA_PAUSE_MIN", 5.0), _env_float("EXTRA_PAUSE_MAX", 12.0))

    # Retries
    max_retries: int = _env_int("MAX_RETRIES", 2)
    backoff_base_sec: float = _env_float("BACKOFF_BASE_SEC", 1.2)
    backoff_max_sec: float = _env_float("BACKOFF_MAX_SEC", 10.0)

    # Quality gates
    require_jobs_path: bool = _env_bool("REQUIRE_JOBS_PATH", True)
    min_description_len: int = _env_int("MIN_DESC_LEN", 250)
    bad_markers: List[str] = field(default_factory=lambda: [
        "jobs based on your profile",
        "sign up for event reminders",
        "jobs in your collections",
        "company ratings",
    ])

    # Evidence / sampling (for debugging & interview artifacts)
    artifact_dir: Path = Path(os.getenv("ARTIFACT_DIR", "data/artifacts/scrape"))
    save_bad_samples: bool = _env_bool("SAVE_BAD_SAMPLES", True)
    bad_sample_max: int = _env_int("BAD_SAMPLE_MAX", 25)  # cap per run

    # Observability
    log_json: bool = _env_bool("LOG_JSON", True)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # SLO checks (example thresholds)
    slo_success_rate: float = _env_float("SLO_SUCCESS_RATE", 0.95)

    # Data Quality SLOs (tune after a few runs)
    dq_desc_len_p50_min: int = _env_int("DQ_DESC_LEN_P50_MIN", 600)
    dq_company_nonnull_rate_min: float = _env_float("DQ_COMPANY_RATE_MIN", 0.85)
    dq_skills_per_job_p50_min: float = _env_float("DQ_SKILLS_P50_MIN", 2.0)