from __future__ import annotations

import time
from typing import Any, Dict, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.services.job_fetch_service import JobFetchService
from src.services.extraction_service import ExtractionService
from src.services.qc_service import QCService
from src.services.report_service import ReportService

# Path("data/artifacts/_debug").mkdir(parents=True, exist_ok=True)


class GraphState(TypedDict, total=False):
    # inputs
    job_id: str
    source: str
    resume_text: Optional[str]

    # provider config
    extract_provider: Literal["openai", "nvidia"]
    extract_model: Optional[str]
    report_provider: Literal["openai", "nvidia"]
    report_model: Optional[str]
    prompt_name: str

    # local config
    local_first: bool
    local_mode: str
    local_model: str
    local_lora_path: Optional[str]

    # runtime handles (not for persistence)
    _mcp_session: Any

    # artifacts
    jd_text: str
    structured: Optional[dict]
    extract_meta: dict
    qc: dict
    report_md: Optional[str]
    report_meta: dict

    # trace
    trace: list[dict]

    # runtime metrics
    metrics: Dict[str, Any]
    decisions: list[dict]
    run_id: str
    ts_start_utc: str


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _trace(state: GraphState, step: str, status: str, **meta: Any) -> None:
    state.setdefault("trace", [])
    state.setdefault("metrics", {}).setdefault("node_ms", {})
    entry = {"t": _now(), "step": step, "status": status, **meta}
    state["trace"].append(entry)


def _tool_result_text(res: Any) -> str:
    parts = []
    for b in getattr(res, "content", []) or []:
        t = getattr(b, "text", None)
        if isinstance(t, str):
            parts.append(t)
    return "".join(parts)



# -----------------------
# Nodes
# -----------------------

async def node_fetch_jd(state: GraphState) -> GraphState:
    started = time.perf_counter()

    job_id = state["job_id"]
    source = state.get("source", "handshake")

    svc = JobFetchService()
    result = await svc.fetch(job_id=job_id, source=source)
    payload = result.to_dict()

    state["jd_text"] = payload["jd_text"]

    trace = list(state.get("trace", []))
    trace.append(
        {
            "node": "fetch_jd",
            "ok": True,
            "source": source,
            "job_id": job_id,
            "jd_len": len(payload["jd_text"]),
            "meta": payload.get("meta") or {},
        }
    )
    state["trace"] = trace

    metrics = dict(state.get("metrics", {}))
    node_ms = dict(metrics.get("node_ms", {}))
    node_ms["fetch_jd"] = round((time.perf_counter() - started) * 1000, 2)
    metrics["node_ms"] = node_ms
    state["metrics"] = metrics

    return state


async def node_extract_local(state: GraphState) -> GraphState:
    started = time.perf_counter()

    jd_text = state["jd_text"]
    job_id = state["job_id"]

    svc = ExtractionService()

    result = await svc.extract_local(
        job_id=job_id,
        jd_text=jd_text,
        prompt_name=state.get("prompt_name", "jd_extract_v2"),
        model=state.get("local_model"),
        lora_path=state.get("local_lora_path"),
        mode=state.get("local_mode", "plain"),
        temperature=state.get("temperature", 0.0),
        max_tokens=state.get("max_tokens", 1024),
    )

    payload = result.to_dict()

    state["structured"] = payload["structured"]
    state["raw_output"] = payload["raw_output"]
    state["parse_ok"] = payload["parse_ok"]
    state["parse_repaired"] = payload["parse_repaired"]
    state["extract_meta"] = payload["extractor"]

    trace = list(state.get("trace", []))
    trace.append(
        {
            "node": "extract_local",
            "ok": payload["parse_ok"],
            "parse_repaired": payload["parse_repaired"],
            "extractor": payload["extractor"],
        }
    )
    state["trace"] = trace

    metrics = dict(state.get("metrics", {}))
    node_ms = dict(metrics.get("node_ms", {}))
    node_ms["extract_local"] = round((time.perf_counter() - started) * 1000, 2)
    metrics["node_ms"] = node_ms
    state["metrics"] = metrics

    return state


async def node_extract_api(state: GraphState) -> GraphState:
    started = time.perf_counter()
    provider = state.get("extract_provider", "openai")

    _trace(state, "extract_api", "start", provider=provider)

    try:
        svc = ExtractionService()
        result = await svc.extract_api(
            job_id=state["job_id"],
            jd_text=state["jd_text"],
            prompt_name=state.get("prompt_name", "jd_extract_v2"),
            provider=provider,
            model=state.get("extract_model"),
            temperature=0.0,
            max_tokens=1200,
        )

        payload = result.to_dict()

        state["structured"] = payload.get("structured")
        state["raw_output"] = payload.get("raw_output")
        state["parse_ok"] = payload.get("parse_ok")
        state["parse_repaired"] = payload.get("parse_repaired")
        state["extract_meta"] = {
            k: payload.get(k) for k in ["parse_ok", "parse_repaired", "usage", "extractor"]
        }

        dt = int((time.perf_counter() - started) * 1000)
        state["metrics"]["node_ms"]["extract_api"] = dt
        _trace(
            state,
            "extract_api",
            "ok",
            parse_ok=payload.get("parse_ok"),
            parse_repaired=payload.get("parse_repaired"),
            elapsed_ms=dt,
        )
        return state

    except Exception as e:
        state["structured"] = None
        state["raw_output"] = ""
        state["parse_ok"] = False
        state["parse_repaired"] = False
        state["extract_meta"] = {
            "parse_ok": False,
            "parse_repaired": False,
            "extractor": {
                "provider": provider,
                "model": state.get("extract_model"),
            },
            "error": str(e),
        }

        dt = int((time.perf_counter() - started) * 1000)
        state["metrics"]["node_ms"]["extract_api"] = dt
        _trace(state, "extract_api", "fail", error=str(e), elapsed_ms=dt)
        return state


async def node_qc(state: GraphState) -> GraphState:
    started = time.perf_counter()

    svc = QCService()

    qc_result = await svc.validate(
        job_id=state["job_id"],
        structured=state.get("structured"),
        parse_ok=bool(state.get("parse_ok")),
        parse_repaired=bool(state.get("parse_repaired")),
        extractor=(state.get("extract_meta") or {}).get("extractor"),
        require_keys=state.get("require_keys", []),
        require_non_empty_any_of=state.get("require_non_empty_any_of", []),
    )

    payload = qc_result.to_dict()
    state["qc"] = payload

    trace = list(state.get("trace", []))
    trace.append(
        {
            "node": "qc",
            "ok": payload["ok"],
            "status": payload["status"],
            "reasons": payload["reasons"],
        }
    )
    state["trace"] = trace

    metrics = dict(state.get("metrics", {}))
    node_ms = dict(metrics.get("node_ms", {}))
    node_ms["qc"] = round((time.perf_counter() - started) * 1000, 2)
    metrics["node_ms"] = node_ms
    state["metrics"] = metrics

    return state


async def node_report(state: GraphState) -> GraphState:
    started = time.perf_counter()

    svc = ReportService()

    result = await svc.generate(
        job_id=state["job_id"],
        structured=state.get("structured") or {},
        qc=state.get("qc") or {},
        match=None,
        resume_text=state.get("resume_text"),
        provider=state.get("report_provider", "openai"),
        model=state.get("report_model"),
        temperature=0.2,
        max_tokens=900,
    )

    payload = result.to_dict()

    state["report_md"] = payload["report_md"]
    state["report_meta"] = {
        "meta": payload.get("meta") or {},
        "usage": payload.get("usage") or {},
    }

    trace = list(state.get("trace", []))
    trace.append(
        {
            "node": "report",
            "ok": bool(payload["report_md"].strip()),
            "provider": state.get("report_provider", "openai"),
            "model": state.get("report_model"),
        }
    )
    state["trace"] = trace

    metrics = dict(state.get("metrics", {}))
    node_ms = dict(metrics.get("node_ms", {}))
    node_ms["report"] = round((time.perf_counter() - started) * 1000, 2)
    metrics["node_ms"] = node_ms
    state["metrics"] = metrics

    return state

async def node_finalize(state: GraphState) -> GraphState:
    _trace(state, "finalize", "ok")
    return state


# -----------------------
# Routers
# -----------------------

def route_after_fetch(state: GraphState) -> str:
    return "extract_local" if state.get("local_first") is True else "extract_api"



def route_after_qc(state: GraphState) -> str:
    qc = state.get("qc", {})
    if qc.get("status") == "pass":
        state.setdefault("decisions", []).append({
            "decision": "qc_pass",
            "at": _now(),
        })
        return "report"

    extractor = (state.get("extract_meta") or {}).get("extractor") or {}

    if extractor.get("provider") in ("openai", "nvidia"):
        state.setdefault("decisions", []).append({
            "decision": "qc_fail_after_api",
            "at": _now(),
            "issues": qc.get("issues"),
        })
        return "finalize"

    state.setdefault("decisions", []).append({
        "decision": "fallback_to_api",
        "at": _now(),
        "reason": "qc_fail_local",
        "issues": qc.get("issues"),
    })
    return "extract_api"

"""
LangGraph is the primary runtime orchestrator for JobPulse.

This graph calls core services directly and does not depend on MCP tool execution.
"""

def build_graph() -> Any:
    g = StateGraph(GraphState)

    g.add_node("fetch", node_fetch_jd)
    g.add_node("extract_local", node_extract_local)
    g.add_node("extract_api", node_extract_api)
    g.add_node("qc", node_qc)
    g.add_node("report", node_report)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("fetch")

    g.add_conditional_edges("fetch", route_after_fetch, {
        "extract_local": "extract_local",
        "extract_api": "extract_api",
    })

    g.add_edge("extract_local", "qc")
    g.add_edge("extract_api", "qc")

    g.add_conditional_edges("qc", route_after_qc, {
        "report": "report",
        "extract_api": "extract_api",
        "finalize": "finalize",
    })

    g.add_edge("report", "finalize")
    g.add_edge("finalize", END)

    return g.compile()
