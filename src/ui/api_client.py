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

def match_resume(resume_text: str, top_k: int = 5) -> dict:
    payload = {"resume_text": resume_text, "top_k": top_k}
    r = requests.post(
        f"{API_BASE}/resume/match",
        json=payload,
        timeout=90,
    )

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = {"raw_text": r.text}
        raise RuntimeError(f"/resume/match failed: status={r.status_code}, detail={detail}")

    return r.json()


def analyze_resume_fit(
    resume_text: str,
    job_id: str,
    include_market_context: bool = True,
    market_top_k: int = 5,
    include_report: bool = True,
    analysis_mode: str = "baseline",
    provider: str = "openai",
    model: str | None = None,
) -> dict:
    payload = {
        "resume_text": resume_text,
        "job_id": job_id,
        "include_market_context": include_market_context,
        "market_top_k": market_top_k,
        "include_report": include_report,
        "analysis_mode": analysis_mode,
        "provider": provider,
        "model": model,
    }

    r = requests.post(
        f"{API_BASE}/resume/analyze-fit",
        json=payload,
        timeout=180,
    )

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = {"raw_text": r.text}
        raise RuntimeError(
            f"/resume/analyze-fit failed: status={r.status_code}, detail={detail}"
        )

    return r.json()


def parse_resume_file(filename: str, file_bytes: bytes) -> dict:
    files = {
        "file": (filename, file_bytes),
    }
    r = requests.post(f"{API_BASE}/resume/parse", files=files, timeout=90)
    
    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = {"raw_text": r.text}
        raise RuntimeError(f"/resume/parse failed: status={r.status_code}, detail={detail}")

    return r.json()

def job_market_chat(
    question: str,
    top_k: int = 5,
    resume_text: str | None = None,
    job_id: str | None = None,
    provider: str = "openai",
    model: str | None = None,
) -> dict:
    payload = {
        "question": question,
        "top_k": top_k,
        "resume_text": resume_text,
        "job_id": job_id,
        "provider": provider,
        "model": model,
    }

    r = requests.post(
        f"{API_BASE}/chat/job-market",
        json=payload,
        timeout=180,
    )

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = {"raw_text": r.text}
        raise RuntimeError(
            f"/chat/job-market failed: status={r.status_code}, detail={detail}"
        )

    return r.json()


def analyze_skill_gap_serverless(
    target_role: str,
    experience_level: str,
    candidate_background: str,
    api_base: str | None = None,
) -> dict:
    base = api_base or os.getenv("JOBPULSE_SERVERLESS_API_BASE")

    if not base:
        raise RuntimeError("Missing JOBPULSE_SERVERLESS_API_BASE env var")

    payload = {
        "target_role": target_role,
        "experience_level": experience_level,
        "candidate_background": candidate_background,
    }

    r = requests.post(
        f"{base}/career/analyze-skill-gap",
        json=payload,
        timeout=60,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
    )

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = {"raw_text": r.text}
        raise RuntimeError(
            f"serverless skill gap failed: status={r.status_code}, detail={detail}"
        )

    return r.json()


def start_career_session() -> dict:
    base = os.getenv("JOBPULSE_SERVERLESS_API_BASE")
    if not base:
        raise RuntimeError("Missing JOBPULSE_SERVERLESS_API_BASE env var")

    r = requests.post(
        f"{base}/career/session/start",
        json={},
        timeout=30,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
    )

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = {"raw_text": r.text}
        raise RuntimeError(
            f"/career/session/start failed: status={r.status_code}, detail={detail}"
        )

    return r.json()


def send_career_message(
    session_id: str,
    message: str,
    resume_text: str | None = None,
) -> dict:
    base = os.getenv("JOBPULSE_SERVERLESS_API_BASE")
    if not base:
        raise RuntimeError("Missing JOBPULSE_SERVERLESS_API_BASE env var")

    payload = {
        "session_id": session_id,
        "message": message,
    }

    if resume_text:
        payload["resume_text"] = resume_text

    r = requests.post(
        f"{base}/career/session/message",
        json=payload,
        timeout=60,
    )

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = {"raw_text": r.text}
        raise RuntimeError(
            f"/career/session/message failed: status={r.status_code}, detail={detail}"
        )

    return r.json()


def get_career_session(session_id: str) -> dict:
    base = os.getenv("JOBPULSE_SERVERLESS_API_BASE")
    if not base:
        raise RuntimeError("Missing JOBPULSE_SERVERLESS_API_BASE env var")

    r = requests.get(
        f"{base}/career/session/{session_id}",
        timeout=30,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
    )

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = {"raw_text": r.text}
        raise RuntimeError(
            f"/career/session/{session_id} failed: status={r.status_code}, detail={detail}"
        )

    return r.json()