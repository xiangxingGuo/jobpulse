from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from src.observability.artifact_writer import JobRunArtifactWriter
from src.orch.graph import build_graph

ARTIFACTS_DIR = Path(os.getenv("ARTIFACT_DIR", "data/artifacts")) / "langgraph"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


async def main_async(args: argparse.Namespace) -> None:
    t0 = time.time()
    app = build_graph()
    run_id = uuid.uuid4().hex[:10]

    started_at = _now()

    state: Dict[str, Any] = {
        # ---- legacy flat fields (temporary compatibility)
        "job_id": args.job_id,
        "source": args.source,
        "prompt_name": args.prompt_name,
        "local_first": args.local_first,
        "local_mode": args.local_mode,
        "local_model": args.local_model,
        "local_lora_path": args.local_lora_path,
        "extract_provider": args.extract_provider,
        "extract_model": args.extract_model,
        "report_provider": args.report_provider,
        "report_model": args.report_model,
        "resume_text": None,
        "trace": [],
        "run_id": run_id,
        "metrics": {"node_ms": {}, "route": None},
        "decisions": [],
        "ts_start_utc": started_at,
        # ---- v2 structured state
        "run": {
            "run_id": run_id,
            "workflow": "job_enrichment_v2",
            "status": "running",
            "started_at": started_at,
            "ended_at": None,
            "route": None,
            "entrypoint": "langgraph",
        },
        "input": {
            "job_id": args.job_id,
            "source": args.source,
            "resume_text": None,
        },
        "config_routing": {
            "primary_mode": "local" if args.local_first else "api",
            "fallback_mode": "api",
            "fallback_enabled": True,
            "local_enabled": bool(args.local_first),
        },
        "config_models": {
            "prompt_name": args.prompt_name,
            "extract_provider": args.extract_provider,
            "extract_model": args.extract_model,
            "report_provider": args.report_provider,
            "report_model": args.report_model,
            "local_mode": args.local_mode,
            "local_model": args.local_model,
            "local_lora_path": args.local_lora_path,
        },
        "qc_policy": {
            "require_keys": ["role_title", "company", "requirements", "responsibilities"],
            "require_non_empty_any_of": [["requirements", "responsibilities"]],
        },
        "features": {
            "enable_skill_gap": False,
        },
    }

    out = await app.ainvoke(state)

    job_dir = Path(args.out_dir) / run_id / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    elapsed = time.time() - t0
    run_meta = out.get("run", {}) or {}
    input_meta = out.get("input", {}) or {}
    # job_meta = out.get("job", {}) or {}
    route = run_meta.get("route") or "unknown"
    decisions = out.get("decisions", [])

    extraction_state = out.get("extraction", {}) or {}
    qc_state = out.get("qc_state", {}) or {}
    # report_state = out.get("report_state", {}) or {}

    legacy_qc = out.get("qc") or {}

    decisions = out.get("decisions", [])

    selected_qc_idx = qc_state.get("selected_attempt")
    qc_attempts = qc_state.get("attempts", [])

    selected_qc = {}
    if isinstance(selected_qc_idx, int) and 0 <= selected_qc_idx < len(qc_attempts):
        selected_qc = qc_attempts[selected_qc_idx]
    elif qc_attempts:
        selected_qc = qc_attempts[-1]
    else:
        selected_qc = legacy_qc

    qc_status = selected_qc.get("status")

    summary = {
        "run_id": run_meta.get("run_id", run_id),
        "job_id": (input_meta or {}).get("job_id", args.job_id),
        "workflow": run_meta.get("workflow", "job_enrichment_v2"),
        "status": run_meta.get("status", "completed"),
        "route": route,
        "started_at": run_meta.get("started_at"),
        "ended_at": run_meta.get("ended_at"),
        "elapsed_sec": round(elapsed, 3),
        "qc_status": qc_status,
        "extraction_attempt_count": len(extraction_state.get("attempts", [])),
        "qc_attempt_count": len(qc_state.get("attempts", [])),
        "node_ms": out.get("metrics", {}).get("node_ms", {}),
        "decisions": decisions,
        "slo": {"availability_pass": qc_status == "pass"},
    }

    selected_extract_idx = extraction_state.get("selected_attempt")
    extract_attempts = extraction_state.get("attempts", [])

    selected_extract = {}
    if isinstance(selected_extract_idx, int) and 0 <= selected_extract_idx < len(extract_attempts):
        selected_extract = extract_attempts[selected_extract_idx]
    elif extract_attempts:
        selected_extract = extract_attempts[-1]
    else:
        selected_extract = {}

    structured_for_legacy = selected_extract.get("structured")
    if structured_for_legacy is None:
        structured_for_legacy = out.get("structured")

    extract_meta_for_legacy = {
        "parse_ok": selected_extract.get("parse_ok"),
        "parse_repaired": selected_extract.get("parse_repaired"),
        "usage": selected_extract.get("usage", {}) or {},
        "extractor": selected_extract.get("extractor", {}) or {},
    }
    if not extract_meta_for_legacy["extractor"]:
        extract_meta_for_legacy = out.get("extract_meta", {})

    writer = JobRunArtifactWriter(args.out_dir)
    job_dir = writer.write(
        run_id=run_id,
        job_id=args.job_id,
        state=out,
        summary=summary,
    )

    print(f"✅ Wrote artifacts to: {job_dir}")
    print(f"[ok] route={route} qc={qc_status} elapsed={round(elapsed, 3)}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--source", default="handshake")
    ap.add_argument("--prompt-name", default="jd_extract_v2")
    ap.add_argument("--out-dir", default=str(ARTIFACTS_DIR))

    # local
    ap.add_argument("--local-first", action="store_true")
    ap.add_argument("--local-mode", default="chat_lora")
    ap.add_argument("--local-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--local-lora-path", default=None)

    # api
    ap.add_argument("--extract-provider", default="openai", choices=["openai", "nvidia"])
    ap.add_argument("--extract-model", default=None)
    ap.add_argument("--report-provider", default="openai", choices=["openai", "nvidia"])
    ap.add_argument("--report-model", default=None)

    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
