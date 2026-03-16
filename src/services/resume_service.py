from __future__ import annotations

import re
from typing import Iterable

from src.schemas.skill_gap import ResumeProfile


# ------------------------------------------------------------
# Canonical skill sets (baseline extraction)
# ------------------------------------------------------------

CANONICAL_SKILLS = {
    "python",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "sklearn",
    "pandas",
    "numpy",
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "azure",
    "sql",
    "spark",
    "airflow",
    "langchain",
    "llm",
    "transformers",
    "huggingface",
    "fastapi",
    "flask",
}

ML_DOMAINS = {
    "nlp",
    "computer vision",
    "cv",
    "speech",
    "recommendation",
    "recommender",
    "rag",
    "retrieval",
    "embedding",
}

DEPLOYMENT_SIGNALS = {
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "azure",
    "deployment",
    "serving",
    "api",
    "fastapi",
    "flask",
}

RESEARCH_SIGNALS = {
    "paper",
    "publication",
    "research",
    "thesis",
    "lab",
    "experiment",
}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return text.lower()


def _find_terms(text: str, terms: Iterable[str]) -> list[str]:
    """Return list of terms found in text using word boundary matching."""
    found = []
    for term in terms:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, text):
            found.append(term)
    return sorted(set(found))


def _preview(text: str, max_chars: int = 600) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ------------------------------------------------------------
# Resume Service
# ------------------------------------------------------------

class ResumeService:
    """
    Responsible for converting raw resume text into a structured ResumeProfile.

    v1 design goals:
    - deterministic baseline extraction
    - no LLM dependency required
    - structured output compatible with SkillGapService
    """

    # ------------------------------
    # Public API
    # ------------------------------

    def parse_profile(self, resume_text: str) -> ResumeProfile:
        """
        Main entrypoint.

        Args:
            resume_text: raw resume text

        Returns:
            ResumeProfile
        """

        if not resume_text or not resume_text.strip():
            raise ValueError("resume_text is empty")

        normalized = _normalize_text(resume_text)

        explicit_skills = _find_terms(normalized, CANONICAL_SKILLS)
        ml_domains = _find_terms(normalized, ML_DOMAINS)
        deployment = _find_terms(normalized, DEPLOYMENT_SIGNALS)
        research = _find_terms(normalized, RESEARCH_SIGNALS)

        profile = ResumeProfile(
            explicit_skills=explicit_skills,
            inferred_skills=[],
            tools=[],
            ml_domains=ml_domains,
            deployment_signals=deployment,
            research_signals=research,
            projects=[],
            evidence_spans=[],
            text_preview=_preview(resume_text),
        )

        return profile

    # ------------------------------
    # Future extension (LLM)
    # ------------------------------

    def enrich_profile_with_llm(
        self,
        profile: ResumeProfile,
        resume_text: str,
    ) -> ResumeProfile:
        """
        Optional enrichment step.

        v1 implementation: no-op.
        Later versions may:
        - infer transferable skills
        - extract project structure
        - detect deployment / ML pipeline signals
        """

        return profile