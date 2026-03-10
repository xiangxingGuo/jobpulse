from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class EmbeddingDocument:
    job_id: str
    text: str
    metadata: dict[str, Any]

def build_embedding_text(job: dict[str, Any]) -> str:
    structured = job.get("structured") or {}
    skills = job.get("skills") or []

    title = structured.get("role_title") or job.get("title") or ""
    company = structured.get("company") or job.get("company") or ""
    location = structured.get("location") or job.get("location_text") or ""

    requirements = structured.get("requirements") or ""
    responsibilities = structured.get("responsibilities") or ""
    years_exp = structured.get("years_experience_min")
    raw_desc = job.get("description") or ""

    skills_text = ", ".join(skills)

    parts = [
        f"Title: {title}",
        f"Company: {company}",
        f"Location: {location}",
    ]

    if years_exp is not None:
        parts.append(f"Years Experience Minimum: {years_exp}")
    if responsibilities:
        parts.append(f"Responsibilities:\n{responsibilities}")
    if requirements:
        parts.append(f"Requirements:\n{requirements}")
    if skills_text:
        parts.append(f"Skills:\n{skills_text}")
    if raw_desc:
        parts.append(f"Raw Description:\n{raw_desc[:4000]}")

    return "\n\n".join([p for p in parts if p.strip()])

def build_documents(rows: list[dict[str, Any]]) -> list[EmbeddingDocument]:
    docs = []
    for row in rows:
        docs.append(
            EmbeddingDocument(
                job_id=row["job_id"],
                text=build_embedding_text(row),
                metadata={
                    "job_id": row["job_id"],
                    "title": row.get("title"),
                    "company": row.get("company"),
                    "location": row.get("location_text"),
                    "url": row.get("url"),
                    "content_hash": row.get("content_hash"),
                },
            )
        )
    return docs