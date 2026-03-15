from __future__ import annotations

import time
from typing import Any, Dict, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.services.job_fetch_service import JobFetchService
from src.services.extraction_service import ExtractionService
from src.services.qc_service import QCService
from src.services.report_service import ReportService

# Path("data/artifacts/_debug").mkdir(parents=True, exist_ok=True)

def _input_value(state: GraphState, key: str, default: Any = None) -> Any:
    return (state.get("input") or {}).get(key, state.get(key, default))


def _routing_value(state: GraphState, key: str, default: Any = None) -> Any:
    return (state.get("config_routing") or {}).get(key, default)


def _model_value(state: GraphState, key: str, default: Any = None) -> Any:
    return (state.get("config_models") or {}).get(key, state.get(key, default))


def _qc_policy_value(state: GraphState, key: str, default: Any = None) -> Any:
    return (state.get("qc_policy") or {}).get(key, default)


class GraphState(TypedDict, total=False):
    # ----------------------------
    # legacy flat fields (temporary compatibility)
    # ----------------------------
    job_id: str
    source: str
    resume_text: Optional[str]

    extract_provider: Literal["openai", "nvidia"]
    extract_model: Optional[str]
    report_provider: Literal["openai", "nvidia"]
    report_model: Optional[str]
    prompt_name: str

    local_first: bool
    local_mode: str
    local_model: str
    local_lora_path: Optional[str]

    jd_text: str
    structured: Optional[dict]
    raw_output: str
    parse_ok: bool
    parse_repaired: bool
    extract_meta: dict
    qc: dict
    report_md: Optional[str]
    report_meta: dict

    trace: list[dict]
    metrics: Dict[str, Any]
    decisions: list[dict]
    run_id: str
    ts_start_utc: str

    # ----------------------------
    # v2 structured state
    # ----------------------------
    run: Dict[str, Any]
    input: Dict[str, Any]
    config_routing: Dict[str, Any]
    config_models: Dict[str, Any]
    qc_policy: Dict[str, Any]
    features: Dict[str, Any]

    job: Dict[str, Any]
    extraction: Dict[str, Any]
    qc_state: Dict[str, Any]
    report_state: Dict[str, Any]
    artifacts: Dict[str, Any]
    errors: list[dict]

    # optional runtime handles
    _mcp_session: Any


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _trace(state: GraphState, step: str, status: str, **meta: Any) -> None:
    state.setdefault("trace", [])
    state.setdefault("metrics", {}).setdefault("node_ms", {})
    entry = {"t": _now(), "step": step, "status": status, **meta}
    state["trace"].append(entry)


def _ensure_v2_state(state: dict[str, Any]) -> None:
    run = state.setdefault("run", {})
    run.setdefault("run_id", state.get("run_id"))
    run.setdefault("workflow", "job_enrichment_v2")
    run.setdefault("status", "running")
    run.setdefault("started_at", state.get("ts_start_utc") or _now())
    run.setdefault("ended_at", None)
    run.setdefault("route", None)
    run.setdefault("entrypoint", "langgraph")

    input_state = state.setdefault("input", {})
    input_state.setdefault("job_id", state.get("job_id"))
    input_state.setdefault("source", state.get("source", "handshake"))
    input_state.setdefault("resume_text", state.get("resume_text"))

    routing = state.setdefault("config_routing", {})
    routing.setdefault("primary_mode", "local" if state.get("local_first") else "api")
    routing.setdefault("fallback_mode", "api")
    routing.setdefault("fallback_enabled", True)
    routing.setdefault("local_enabled", bool(state.get("local_first")))

    models = state.setdefault("config_models", {})
    models.setdefault("prompt_name", state.get("prompt_name", "jd_extract_v2"))
    models.setdefault("extract_provider", state.get("extract_provider", "openai"))
    models.setdefault("extract_model", state.get("extract_model"))
    models.setdefault("report_provider", state.get("report_provider", "openai"))
    models.setdefault("report_model", state.get("report_model"))
    models.setdefault("local_mode", state.get("local_mode", "chat_lora"))
    models.setdefault("local_model", state.get("local_model", "Qwen/Qwen2.5-0.5B-Instruct"))
    models.setdefault("local_lora_path", state.get("local_lora_path"))

    qc_policy = state.setdefault("qc_policy", {})
    qc_policy.setdefault("require_keys", ["role_title", "company", "requirements", "responsibilities"])
    qc_policy.setdefault("require_non_empty_any_of", [["requirements", "responsibilities"]])

    features = state.setdefault("features", {})
    features.setdefault("enable_skill_gap", False)

    job = state.setdefault("job", {})
    job.setdefault("jd_text", state.get("jd_text"))
    job.setdefault("jd_path", None)
    job.setdefault("jd_meta", {})

    extraction = state.setdefault("extraction", {})
    extraction.setdefault("attempts", [])
    extraction.setdefault("selected_attempt", None)

    qc_state = state.setdefault("qc_state", {})
    qc_state.setdefault("attempts", [])
    qc_state.setdefault("selected_attempt", None)

    report_state = state.setdefault("report_state", {})
    report_state.setdefault("report_md", state.get("report_md"))
    report_state.setdefault("meta", (state.get("report_meta") or {}).get("meta", {}))
    report_state.setdefault("usage", (state.get("report_meta") or {}).get("usage", {}))

    artifacts = state.setdefault("artifacts", {})
    artifacts.setdefault("base_dir", None)
    artifacts.setdefault("paths", {})

    state.setdefault("errors", [])


def _append_extraction_attempt(
    state: dict[str, Any],
    *,
    stage: str,
    mode: str,
    payload: dict[str, Any],
    error: str | None = None,
) -> None:
    _ensure_v2_state(state)

    attempt = {
        "stage": stage,
        "mode": mode,
        "structured": payload.get("structured"),
        "raw_output": payload.get("raw_output", ""),
        "parse_ok": bool(payload.get("parse_ok")),
        "parse_repaired": bool(payload.get("parse_repaired")),
        "extractor": payload.get("extractor", {}),
        "usage": payload.get("usage", {}) or {},
        "error": error,
    }

    state["extraction"]["attempts"].append(attempt)
    state["extraction"]["selected_attempt"] = len(state["extraction"]["attempts"]) - 1


def _append_qc_attempt(
    state: dict[str, Any],
    *,
    stage: str,
    payload: dict[str, Any],
) -> None:
    _ensure_v2_state(state)

    attempt = {
        "stage": stage,
        "ok": bool(payload.get("ok")),
        "status": payload.get("status", "fail"),
        "reasons": payload.get("reasons", []) or [],
        "checks": payload.get("checks", {}) or {},
        "summary": payload.get("summary"),
    }

    state["qc_state"]["attempts"].append(attempt)
    state["qc_state"]["selected_attempt"] = len(state["qc_state"]["attempts"]) - 1


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

    _ensure_v2_state(state)
    job_id = _input_value(state, "job_id")
    source = _input_value(state, "source", "handshake")

    svc = JobFetchService()
    result = await svc.fetch(job_id=job_id, source=source)
    payload = result.to_dict()

    state["jd_text"] = payload["jd_text"]
    state["job"]["jd_text"] = payload["jd_text"]
    state["job"]["jd_path"] = payload.get("jd_path")
    state["job"]["jd_meta"] = payload.get("meta") or {}

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

    svc = ExtractionService()

    result = await svc.extract_local(
        job_id=_input_value(state, "job_id"),
        jd_text=state["jd_text"],
        prompt_name=_model_value(state, "prompt_name", "jd_extract_v2"),
        model=_model_value(state, "local_model"),
        lora_path=_model_value(state, "local_lora_path"),
        mode=_model_value(state, "local_mode", "plain"),
        temperature=state.get("temperature", 0.0),
        max_tokens=state.get("max_tokens", 1024),
    )

    payload = result.to_dict()

    state["structured"] = payload["structured"]
    state["raw_output"] = payload["raw_output"]
    state["parse_ok"] = payload["parse_ok"]
    state["parse_repaired"] = payload["parse_repaired"]
    state["extract_meta"] = {
        "parse_ok": payload["parse_ok"],
        "parse_repaired": payload["parse_repaired"],
        "usage": payload.get("usage") or {},
        "extractor": payload.get("extractor") or {},
    }

    _append_extraction_attempt(
        state,
        stage="primary",
        mode="local",
        payload=payload,
    )

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
    _ensure_v2_state(state)

    try:
        svc = ExtractionService()
        result = await svc.extract_api(
            job_id=_input_value(state, "job_id"),
            jd_text=state["jd_text"],
            prompt_name=_model_value(state, "prompt_name", "jd_extract_v2"),
            provider=provider,
            model=_model_value(state, "extract_model"),
            temperature=0.0,
            max_tokens=1200,
        )
        payload = result.to_dict()

        state["structured"] = payload.get("structured")
        state["raw_output"] = payload.get("raw_output")
        state["parse_ok"] = payload.get("parse_ok")
        state["parse_repaired"] = payload.get("parse_repaired")
        state["extract_meta"] = {
            "parse_ok": payload.get("parse_ok"),
            "parse_repaired": payload.get("parse_repaired"),
            "usage": payload.get("usage") or {},
            "extractor": payload.get("extractor") or {},
        }

        stage = "fallback" if state.get("config_routing", {}).get("primary_mode") == "local" else "primary"

        _append_extraction_attempt(
            state,
            stage=stage,
            mode="api",
            payload=payload,
        )

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
    _ensure_v2_state(state)
    qc_result = await svc.validate(
        job_id=_input_value(state, "job_id"),
        structured=state.get("structured"),
        parse_ok=bool(state.get("parse_ok")),
        parse_repaired=bool(state.get("parse_repaired")),
        extractor=(state.get("extract_meta") or {}).get("extractor"),
        require_keys=_qc_policy_value(state, "require_keys", []),
        require_non_empty_any_of=_qc_policy_value(state, "require_non_empty_any_of", []),
    )

    payload = qc_result.to_dict()
    state["qc"] = payload
    stage = "fallback" if len(state.get("extraction", {}).get("attempts", [])) > 1 else "primary"
    _append_qc_attempt(state, stage=stage, payload=payload)

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

    _ensure_v2_state(state)

    svc = ReportService()


    result = await svc.generate(
        job_id=_input_value(state, "job_id"),
        structured=state.get("structured") or {},
        qc=state.get("qc") or {},
        match=None,
        resume_text=_input_value(state, "resume_text"),
        provider=_model_value(state, "report_provider", "openai"),
        model=_model_value(state, "report_model"),
        temperature=0.2,
        max_tokens=900,
    )

    payload = result.to_dict()

    state["report_md"] = payload["report_md"]
    state["report_meta"] = {
        "meta": payload.get("meta") or {},
        "usage": payload.get("usage") or {},
    }

    state["report_state"]["report_md"] = payload["report_md"]
    state["report_state"]["meta"] = payload.get("meta") or {}
    state["report_state"]["usage"] = payload.get("usage") or {}

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
    _ensure_v2_state(state)

    if not state["run"].get("route"):
        extractor = (state.get("extract_meta") or {}).get("extractor") or {}
        if extractor.get("provider") in ("openai", "nvidia"):
            state["run"]["route"] = "api_only"
        elif extractor.get("mode") in ("plain", "chat_lora"):
            state["run"]["route"] = "local_only"
        else:
            state["run"]["route"] = "unknown"

    state["run"]["status"] = "completed"
    state["run"]["ended_at"] = _now()

    _trace(state, "finalize", "ok", route=state["run"]["route"])
    return state


# -----------------------
# Routers
# -----------------------

def route_after_fetch(state: GraphState) -> str:
    _ensure_v2_state(state)
    primary_mode = _routing_value(state, "primary_mode", "api")
    return "extract_local" if primary_mode == "local" else "extract_api"



def route_after_qc(state: GraphState) -> str:
    _ensure_v2_state(state)

    qc = state.get("qc", {})
    extractor = (state.get("extract_meta") or {}).get("extractor") or {}
    routing = state.get("config_routing", {})

    if qc.get("status") == "pass":
        route = "local_then_report" if len(state.get("extraction", {}).get("attempts", [])) == 1 and extractor.get("mode") in ("plain", "chat_lora") else "api_then_report"
        state["run"]["route"] = route
        state.setdefault("decisions", []).append({
            "decision": "qc_pass",
            "at": _now(),
            "route": route,
        })
        return "report"

    used_api = extractor.get("provider") in ("openai", "nvidia")
    fallback_enabled = bool(_routing_value(state, "fallback_enabled", True))
    primary_mode = _routing_value(state, "primary_mode", "api")

    if used_api or not fallback_enabled or primary_mode == "api":
        state["run"]["route"] = "failed_no_qc_pass"
        state.setdefault("decisions", []).append({
            "decision": "qc_fail_terminal",
            "at": _now(),
            "reasons": qc.get("reasons") or qc.get("issues"),
            "route": state["run"]["route"],
        })
        return "finalize"

    state["run"]["route"] = "local_then_api"
    state.setdefault("decisions", []).append({
        "decision": "fallback_to_api",
        "at": _now(),
        "reason": "qc_fail_local",
        "reasons": qc.get("reasons") or qc.get("issues"),
        "route": state["run"]["route"],
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
