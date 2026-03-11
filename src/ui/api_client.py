from __future__ import annotations

import os
import requests


API_BASE = os.getenv("JOBPULSE_API_BASE", "http://127.0.0.1:8000")


def get_health() -> dict:
    r = requests.get(f"{API_BASE}/health", timeout=30)
    r.raise_for_status()
    return r.json()


def search_jobs(query: str, top_k: int = 10) -> dict:
    r = requests.post(
        f"{API_BASE}/jobs/search",
        json={"query": query, "top_k": top_k},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def get_job(job_id: str) -> dict:
    r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=60)
    r.raise_for_status()
    return r.json()


def get_similar_jobs(job_id: str, top_k: int = 5) -> dict:
    r = requests.get(f"{API_BASE}/jobs/{job_id}/similar?top_k={top_k}", timeout=60)
    r.raise_for_status()
    return r.json()


def get_metrics(limit: int = 20) -> dict:
    r = requests.get(f"{API_BASE}/metrics/summary?limit={limit}", timeout=60)
    r.raise_for_status()
    return r.json()


def get_recent_runs(limit: int = 10) -> dict:
    r = requests.get(f"{API_BASE}/runs/recent?limit={limit}", timeout=60)
    r.raise_for_status()
    return r.json()

def get_analytics_summary(limit: int = 10) -> dict:
    r = requests.get(f"{API_BASE}/analytics/summary?limit={limit}", timeout=60)
    r.raise_for_status()
    return r.json()