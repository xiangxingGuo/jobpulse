from __future__ import annotations

import re
from typing import Any

from src.db import fetch_job_detail
from src.retrieval.search import JobSearchService


CANONICAL_SKILLS = [
    "python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "sql", "docker", "kubernetes", "aws", "gcp", "azure",
    "fastapi", "streamlit", "faiss", "langgraph", "transformers",
    "llm", "rag", "mlops", "airflow", "spark", "pandas",
    "numpy", "linux", "git"
]


def extract_resume_skills(resume_text: str) -> list[str]:
    text = (resume_text or "").lower()
    found = []
    for skill in CANONICAL_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found.append(skill)
    return sorted(set(found))


def build_resume_query(resume_text: str, skills: list[str]) -> str:
    preview = " ".join((resume_text or "").split())[:1200]
    skills_text = ", ".join(skills)
    return f"Resume Skills: {skills_text}\n\nResume Text:\n{preview}"


def match_resume_to_jobs(
    resume_text: str,
    top_k: int = 5,
) -> dict[str, Any]:
    skills = extract_resume_skills(resume_text)
    query = build_resume_query(resume_text, skills)

    svc = JobSearchService()
    semantic_hits = svc.search_jobs(query, top_k=top_k)

    matches = []
    for hit in semantic_hits:
        job_id = str(hit["job_id"])
        detail = fetch_job_detail(job_id)
        job_skills = [s.lower() for s in (detail or {}).get("skills", [])]

        shared = sorted(set(skills) & set(job_skills))
        missing = sorted(set(job_skills) - set(skills))

        reasons = []
        if shared:
            reasons.append(f"Shared skills: {', '.join(shared[:5])}")
        if hit.get("score") is not None:
            reasons.append(f"Strong semantic similarity score: {hit['score']:.3f}")
        if not reasons:
            reasons.append("General semantic alignment with the job description.")

        matches.append(
            {
                "job_id": job_id,
                "title": hit.get("title"),
                "company": hit.get("company"),
                "location": hit.get("location"),
                "url": hit.get("url"),
                "semantic_score": float(hit.get("score", 0.0)),
                "shared_skills": shared,
                "missing_skills": missing[:8],
                "match_reasons": reasons,
            }
        )

    return {
        "resume_profile": {
            "skills": skills,
            "text_preview": " ".join((resume_text or "").split())[:300],
        },
        "matches": matches,
    }
