from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class JobRunArtifactWriter:
    """
    Persist LangGraph run artifacts from the v2 state shape,
    while still writing a few legacy compatibility files.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def write(
        self, *, run_id: str, job_id: str, state: Dict[str, Any], summary: Dict[str, Any]
    ) -> Path:
        job_dir = self.base_dir / run_id / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        run_meta = state.get("run", {}) or {}
        input_meta = state.get("input", {}) or {}
        job_meta = state.get("job", {}) or {}
        extraction_state = state.get("extraction", {}) or {}
        qc_state = state.get("qc_state", {}) or {}
        report_state = state.get("report_state", {}) or {}
        trace = state.get("trace", []) or []

        # ----------------------------
        # v2 artifacts
        # ----------------------------
        _write_json(job_dir / "run_summary.json", summary)
        _write_json(job_dir / "run.json", run_meta)
        _write_json(job_dir / "input.json", input_meta)
        _write_json(job_dir / "job.json", job_meta)
        _write_json(job_dir / "extraction_attempts.json", extraction_state.get("attempts", []))
        _write_json(job_dir / "qc_attempts.json", qc_state.get("attempts", []))
        _write_json(job_dir / "report_state.json", report_state)
        _write_json(job_dir / "trace.json", trace)

        # ----------------------------
        # selected extraction / qc
        # ----------------------------
        extract_attempts = extraction_state.get("attempts", []) or []
        selected_extract_idx = extraction_state.get("selected_attempt")

        selected_extract: Dict[str, Any] = {}
        if isinstance(selected_extract_idx, int) and 0 <= selected_extract_idx < len(
            extract_attempts
        ):
            selected_extract = extract_attempts[selected_extract_idx]
        elif extract_attempts:
            selected_extract = extract_attempts[-1]

        qc_attempts = qc_state.get("attempts", []) or []
        selected_qc_idx = qc_state.get("selected_attempt")

        selected_qc: Dict[str, Any] = {}
        if isinstance(selected_qc_idx, int) and 0 <= selected_qc_idx < len(qc_attempts):
            selected_qc = qc_attempts[selected_qc_idx]
        elif qc_attempts:
            selected_qc = qc_attempts[-1]
        else:
            selected_qc = state.get("qc", {}) or {}

        report_md = report_state.get("report_md") or state.get("report_md") or ""
        report_meta = {
            "meta": report_state.get("meta", {}) or {},
            "usage": report_state.get("usage", {}) or {},
        }
        if not report_meta["meta"] and not report_meta["usage"]:
            report_meta = state.get("report_meta", {}) or {}

        structured_for_legacy = selected_extract.get("structured")
        if structured_for_legacy is None:
            structured_for_legacy = state.get("structured")

        extract_meta_for_legacy = {
            "parse_ok": selected_extract.get("parse_ok"),
            "parse_repaired": selected_extract.get("parse_repaired"),
            "usage": selected_extract.get("usage", {}) or {},
            "extractor": selected_extract.get("extractor", {}) or {},
        }
        if not extract_meta_for_legacy["extractor"]:
            extract_meta_for_legacy = state.get("extract_meta", {}) or {}

        jd_text = job_meta.get("jd_text") or state.get("jd_text", "")

        # ----------------------------
        # legacy compatibility artifacts
        # ----------------------------
        _write_text(job_dir / "jd.txt", jd_text)
        _write_json(job_dir / "structured.json", structured_for_legacy)
        _write_json(job_dir / "extract_meta.json", extract_meta_for_legacy)
        _write_json(job_dir / "qc.json", selected_qc)
        _write_text(job_dir / "report.md", report_md)
        _write_json(job_dir / "report_meta.json", report_meta)

        return job_dir


class SkillGapArtifactWriter:
    """
    Persist artifacts for resume-to-job skill-gap analysis.

    Intended for API/UI-triggered analysis flows, independent of LangGraph.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def write(
        self,
        *,
        run_id: str,
        job_id: str,
        resume_profile: Dict[str, Any],
        skill_gap: Dict[str, Any],
        report_md: str = "",
        meta: Dict[str, Any] | None = None,
        artifacts: Dict[str, Any] | None = None,
        resume_text: str | None = None,
        structured_job: Dict[str, Any] | None = None,
    ) -> Path:
        job_dir = self.base_dir / run_id / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        artifacts = artifacts or {}
        meta = meta or {}
        structured_job = structured_job or {}

        _write_json(job_dir / "resume_profile.json", resume_profile)
        _write_json(job_dir / "skill_gap.json", skill_gap)
        _write_text(job_dir / "skill_gap_report.md", report_md)
        _write_json(job_dir / "analysis_meta.json", meta)

        if resume_text is not None:
            _write_text(job_dir / "resume.txt", resume_text)

        if structured_job:
            _write_json(job_dir / "job.json", structured_job)

        baseline = artifacts.get("baseline")
        if baseline is not None:
            _write_json(job_dir / "baseline.json", baseline)

        job_context = artifacts.get("job_context")
        if job_context is not None:
            _write_json(job_dir / "job_context.json", job_context)

        market_context = artifacts.get("market_context")
        if market_context is not None:
            _write_json(job_dir / "market_context.json", market_context)

        run_summary = {
            "run_id": run_id,
            "job_id": job_id,
            "workflow": "skill_gap_analysis_v1",
            "has_report": bool((report_md or "").strip()),
            "fit_score": skill_gap.get("fit_score"),
            "fit_band": skill_gap.get("fit_band"),
            "strength_count": len(skill_gap.get("strengths", []) or []),
            "gap_count": len(skill_gap.get("gaps", []) or []),
            "transferable_count": len(skill_gap.get("transferable_signals", []) or []),
        }
        _write_json(job_dir / "run_summary.json", run_summary)

        return job_dir


class JobMarketChatArtifactWriter:
    """
    Persist artifacts for single-turn grounded job-market chat.

    Intended for API/UI-triggered chat flows, independent of LangGraph.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def write(
        self,
        *,
        run_id: str,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        meta: Dict[str, Any] | None = None,
        artifacts: Dict[str, Any] | None = None,
        resume_text: str | None = None,
        job_id: str | None = None,
    ) -> Path:
        chat_dir = self.base_dir / run_id
        chat_dir.mkdir(parents=True, exist_ok=True)

        meta = meta or {}
        artifacts = artifacts or {}

        _write_text(chat_dir / "question.txt", question or "")
        _write_text(chat_dir / "answer.md", answer or "")
        _write_json(chat_dir / "sources.json", sources or [])
        _write_json(chat_dir / "meta.json", meta)

        if resume_text is not None:
            _write_text(chat_dir / "resume.txt", resume_text)

        if job_id:
            _write_text(chat_dir / "job_id.txt", str(job_id))

        retrieved_jobs = artifacts.get("retrieved_jobs")
        if retrieved_jobs is not None:
            _write_json(chat_dir / "retrieved_jobs.json", retrieved_jobs)

        target_job = artifacts.get("target_job")
        if target_job is not None:
            _write_json(chat_dir / "target_job.json", target_job)

        resume_profile = artifacts.get("resume_profile")
        if resume_profile is not None:
            _write_json(chat_dir / "resume_profile.json", resume_profile)

        skill_gap = artifacts.get("skill_gap")
        if skill_gap is not None:
            _write_json(chat_dir / "skill_gap.json", skill_gap)

        llm = artifacts.get("llm")
        if llm is not None:
            _write_json(chat_dir / "llm.json", llm)

            raw_output = llm.get("raw_output")
            if raw_output is not None:
                _write_text(chat_dir / "llm_raw_output.txt", str(raw_output))

        run_summary = {
            "run_id": run_id,
            "workflow": "job_market_chat_v1",
            "job_id": job_id,
            "question_chars": len((question or "").strip()),
            "answer_chars": len((answer or "").strip()),
            "source_count": len(sources or []),
            "used_resume": bool((resume_text or "").strip()),
            "used_target_job": bool(job_id),
            "provider": meta.get("provider"),
            "model": meta.get("model"),
        }
        _write_json(chat_dir / "run_summary.json", run_summary)

        return chat_dir
