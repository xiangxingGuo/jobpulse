from __future__ import annotations

import json
from typing import Any, Dict, List


def build_job_market_chat_messages(
    *,
    question: str,
    retrieved_jobs: List[Dict[str, Any]],
    resume_profile: Dict[str, Any] | None = None,
    skill_gap: Dict[str, Any] | None = None,
    target_job: Dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    resume_profile = resume_profile or {}
    skill_gap = skill_gap or {}
    target_job = target_job or {}

    system = (
        "You are a grounded career intelligence assistant.\n"
        "Answer the user's question using ONLY the provided job-market context, target job, "
        "resume profile, and skill-gap analysis if available.\n"
        "Do not invent job requirements, company details, or candidate experience.\n"
        "If the evidence is weak or incomplete, say so.\n"
        "At the end, return a short source list in JSON."
    )

    user = (
        "You are given:\n"
        "1. A user question\n"
        "2. Retrieved relevant jobs from the local job corpus\n"
        "3. Optional target job detail\n"
        "4. Optional resume profile\n"
        "5. Optional skill-gap analysis\n\n"

        "Your task:\n"
        "- answer the user question clearly and practically\n"
        "- stay grounded in the provided context\n"
        "- compare jobs when useful\n"
        "- mention uncertainty when evidence is limited\n"
        "- do not fabricate skills or requirements\n\n"

        "Return output in exactly this JSON format:\n"
        "{\n"
        '  "answer": "string",\n'
        '  "sources": [\n'
        "    {\n"
        '      "job_id": "string",\n'
        '      "title": "string",\n'
        '      "company": "string",\n'
        '      "reason": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"

        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED_JOBS_JSON:\n{_pretty(retrieved_jobs)}\n\n"
        f"TARGET_JOB_JSON:\n{_pretty(target_job)}\n\n"
        f"RESUME_PROFILE_JSON:\n{_pretty(resume_profile)}\n\n"
        f"SKILL_GAP_JSON:\n{_pretty(skill_gap)}\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)