from __future__ import annotations

import json
from typing import Any, Dict


def build_skill_gap_analysis_messages(
    *,
    resume_profile: Dict[str, Any],
    resume_text: str,
    job_detail: Dict[str, Any],
    baseline: Dict[str, Any],
    market_context: Dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Build chat-completions style messages for hybrid skill-gap analysis.

    The LLM should:
    - reason over direct evidence vs transferable evidence
    - classify gaps more intelligently than the baseline
    - stay grounded in provided inputs only
    - return strict JSON
    """
    market_context = market_context or {}

    system = (
        "You are a careful career intelligence evaluator.\n"
        "Your task is to compare a candidate resume against a target job.\n"
        "You must use ONLY the provided inputs.\n"
        "Do not invent skills, qualifications, experience, achievements, or job requirements.\n"
        "If evidence is weak, ambiguous, or indirect, say so explicitly.\n"
        "Return STRICT JSON only. No markdown. No prose outside JSON."
    )

    user = (
        "You are given five grounded inputs:\n"
        "1. Resume profile extracted from the candidate resume\n"
        "2. Raw resume text\n"
        "3. Structured target job data\n"
        "4. Baseline overlap signals\n"
        "5. Market context from similar jobs\n\n"

        "Your job:\n"
        "- identify direct strengths\n"
        "- identify transferable strengths\n"
        "- identify must-have gaps, nice-to-have gaps, and ambiguous gaps\n"
        "- suggest grounded resume improvements\n"
        "- write a concise summary\n\n"

        "Important rules:\n"
        "- A direct strength must have clear support from the resume/profile.\n"
        "- A transferable strength may be inferred from adjacent experience, but must say what that adjacent evidence is.\n"
        "- A gap should be marked 'must_have' only when the job strongly suggests it is central.\n"
        "- Do not say the candidate has experience they do not have.\n"
        "- Resume suggestions must improve phrasing or evidence presentation, not fabricate content.\n"
        "- Keep output concise and grounded.\n\n"

        "Return JSON in exactly this shape:\n"
        "{\n"
        '  "strengths": [\n'
        "    {\n"
        '      "skill": "string",\n'
        '      "support": "direct",\n'
        '      "rationale": "string",\n'
        '      "evidence": [\n'
        "        {\n"
        '          "claim": "string",\n'
        '          "source": "resume" | "job" | "market" | "baseline",\n'
        '          "snippet": "string",\n'
        '          "score": 0.0\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "gaps": [\n'
        "    {\n"
        '      "skill": "string",\n'
        '      "category": "must_have" | "nice_to_have" | "ambiguous",\n'
        '      "severity": "high" | "medium" | "low",\n'
        '      "rationale": "string",\n'
        '      "evidence": [\n'
        "        {\n"
        '          "claim": "string",\n'
        '          "source": "resume" | "job" | "market" | "baseline",\n'
        '          "snippet": "string",\n'
        '          "score": 0.0\n'
        "        }\n"
        "      ],\n"
        '      "actionable": true\n'
        "    }\n"
        "  ],\n"
        '  "transferable_signals": [\n'
        "    {\n"
        '      "skill": "string",\n'
        '      "support": "transferable",\n'
        '      "rationale": "string",\n'
        '      "evidence": [\n'
        "        {\n"
        '          "claim": "string",\n'
        '          "source": "resume" | "job" | "market" | "baseline",\n'
        '          "snippet": "string",\n'
        '          "score": 0.0\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "resume_suggestions": [\n'
        "    {\n"
        '      "type": "rewrite" | "add_evidence" | "reorder" | "clarify",\n'
        '      "target": "string",\n'
        '      "before": null,\n'
        '      "after": null,\n'
        '      "rationale": "string"\n'
        "    }\n"
        "  ],\n"
        '  "summary": "string"\n'
        "}\n\n"

        "Guidance on evidence:\n"
        "- Prefer short snippets, not long quotations.\n"
        "- Use source='baseline' when referencing overlap or missing-skill signals.\n"
        "- Use source='market' only when the similar-job context is relevant.\n\n"

        "Inputs below.\n\n"
        f"RESUME_PROFILE_JSON:\n{_to_pretty_json(resume_profile)}\n\n"
        f"RESUME_TEXT:\n{_truncate_text(resume_text, max_chars=4000)}\n\n"
        f"JOB_DETAIL_JSON:\n{_to_pretty_json(job_detail)}\n\n"
        f"BASELINE_JSON:\n{_to_pretty_json(baseline)}\n\n"
        f"MARKET_CONTEXT_JSON:\n{_to_pretty_json(market_context)}\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _to_pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _truncate_text(text: str, max_chars: int = 4000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"