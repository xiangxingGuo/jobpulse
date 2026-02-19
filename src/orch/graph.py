from __future__ import annotations

import os
import time
from typing import Any, Dict, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
from pathlib import Path

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


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _trace(state: GraphState, step: str, status: str, **meta: Any) -> None:
    state.setdefault("trace", [])
    state["trace"].append({"t": _now(), "step": step, "status": status, **meta})


def _tool_result_text(res: Any) -> str:
    parts = []
    for b in getattr(res, "content", []) or []:
        t = getattr(b, "text", None)
        if isinstance(t, str):
            parts.append(t)
    return "".join(parts)


def _tool_result_json(res: Any, tool_name: str) -> Any:
    if getattr(res, "is_error", False):
        raise RuntimeError(f"Tool {tool_name} failed: {_tool_result_text(res).strip()}")
    txt = _tool_result_text(res).strip()
    if not txt:
        raise RuntimeError(f"Tool {tool_name} returned empty content")
    import json
    return json.loads(txt)


async def _call_tool_once(tool_name: str, args: Dict[str, Any]) -> Any:
    """
    Short-lived MCP stdio connection: open -> initialize -> call_tool -> close.
    This avoids cross-task cancel-scope issues with LangGraph async runner.
    """
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp_server.server"],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_name, args)


async def _ensure_session(state: GraphState) -> ClientSession:
    if state.get("_mcp_session") is not None:
        return state["_mcp_session"]  # type: ignore

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp_server.server"],
        env=dict(os.environ),
    )

    # create a live session and store it in state
    read, write = await stdio_client(server_params).__aenter__()  # type: ignore
    session_cm = ClientSession(read, write)
    session = await session_cm.__aenter__()  # type: ignore
    await session.initialize()

    # also store context managers so we can close later
    state["_mcp_session"] = session
    state["_mcp_session_cm"] = session_cm
    state["_mcp_stdio_cm"] = stdio_client(server_params)

    return session


async def _close_session(state: GraphState) -> None:
    # best-effort close
    try:
        sess_cm = state.get("_mcp_session_cm")
        if sess_cm:
            await sess_cm.__aexit__(None, None, None)  # type: ignore
    except Exception:
        pass
    try:
        stdio_cm = state.get("_mcp_stdio_cm")
        if stdio_cm:
            await stdio_cm.__aexit__(None, None, None)  # type: ignore
    except Exception:
        pass


# -----------------------
# Nodes
# -----------------------

async def node_fetch_jd(state: GraphState) -> GraphState:
    _trace(state, "fetch_jd", "start")
    # session = await _ensure_session(state)
    res = await _call_tool_once("fetch_jd", {"job_id": state["job_id"], "source": state.get("source", "handshake")})
    payload = _tool_result_json(res, "fetch_jd")
    state["jd_text"] = payload["jd_text"]
    _trace(state, "fetch_jd", "ok", jd_len=len(state["jd_text"]))
    return state


async def node_extract_local(state: GraphState) -> GraphState:
    _trace(state, "extract_local", "start")
    args: Dict[str, Any] = {
        "job_id": state["job_id"],
        "jd_text": state["jd_text"],
        "prompt_name": state.get("prompt_name", "jd_extract_v2"),
        "model": state.get("local_model", "Qwen/Qwen2.5-0.5B-Instruct"),
        "lora_path": state.get("local_lora_path"),
        "mode": state.get("local_mode", "chat_lora"),
    }

    try:
        res = await _call_tool_once("extract_local", args)

        # Path(f"data/artifacts/_debug/extract_local_res_{state['job_id']}.txt").write_text(repr(res), encoding="utf-8")


        # If tool itself errored, treat as local failure (do not crash graph)
        if getattr(res, "is_error", False):
            err = _tool_result_text(res).strip() or "unknown tool error"
            state["structured"] = None
            state["extract_meta"] = {
                "parse_ok": False,
                "parse_repaired": False,
                "extractor": {"mode": args["mode"], "model": args["model"], "lora_path": args.get("lora_path")},
                "error": f"tool_error: {err}",
            }
            _trace(state, "extract_local", "fail", error=err)
            return state

        txt = _tool_result_text(res).strip()
        if not txt:
            state["structured"] = None
            state["extract_meta"] = {
                "parse_ok": False,
                "parse_repaired": False,
                "extractor": {"mode": args["mode"], "model": args["model"], "lora_path": args.get("lora_path")},
                "error": "empty_tool_content",
            }
            _trace(state, "extract_local", "fail", error="empty_tool_content")
            return state

        import json
        payload = json.loads(txt)

        state["structured"] = payload.get("structured")
        state["extract_meta"] = {k: payload.get(k) for k in ["parse_ok", "parse_repaired", "extractor", "usage"]}
        _trace(state, "extract_local", "ok", parse_ok=payload.get("parse_ok"))
        return state

    except Exception as e:
        # JSON decode errors and any unexpected errors land here
        state["structured"] = None
        state["extract_meta"] = {
            "parse_ok": False,
            "parse_repaired": False,
            "extractor": {"mode": args["mode"], "model": args["model"], "lora_path": args.get("lora_path")},
            "error": str(e),
        }
        _trace(state, "extract_local", "fail", error=str(e))
        return state


async def node_extract_api(state: GraphState) -> GraphState:
    _trace(state, "extract_api", "start", provider=state.get("extract_provider"))
    # session = await _ensure_session(state)

    provider = state.get("extract_provider", "openai")
    model = state.get("extract_model")

    args: Dict[str, Any] = {
        "job_id": state["job_id"],
        "jd_text": state["jd_text"],
        "prompt_name": state.get("prompt_name", "jd_extract_v2"),
        "provider": provider,
        "temperature": 0.0,
        "max_tokens": 1200,
    }
    if model:
        args["model"] = model
    # if provider == "nvidia": args["thinking"] = "disabled"  # only if you kept this param in tool

    res = await _call_tool_once("extract_api", args)
    payload = _tool_result_json(res, "extract_api")

    state["structured"] = payload.get("structured")
    state["extract_meta"] = {k: payload.get(k) for k in ["parse_ok", "parse_repaired", "usage", "extractor"]}
    _trace(state, "extract_api", "ok", parse_ok=payload.get("parse_ok"))
    return state


async def node_qc(state: GraphState) -> GraphState:
    _trace(state, "qc_validate", "start")
    # session = await _ensure_session(state)
    args = {
        "job_id": state["job_id"],
        "structured": state.get("structured"),
        "parse_ok": bool(state.get("extract_meta", {}).get("parse_ok", state.get("structured") is not None)),
        "parse_repaired": bool(state.get("extract_meta", {}).get("parse_repaired", False)),
        "extractor": state.get("extract_meta", {}).get("extractor", {}),
        "require_keys": ["role_title", "company", "requirements", "responsibilities"],
        "require_non_empty_any_of": [["requirements", "responsibilities"]],
    }
    res = await _call_tool_once("qc_validate", args)
    payload = _tool_result_json(res, "qc_validate")
    state["qc"] = payload
    _trace(state, "qc_validate", "ok", qc=payload.get("status"), issues=payload.get("issues"))
    return state


async def node_report(state: GraphState) -> GraphState:
    _trace(state, "generate_report_api", "start", provider=state.get("report_provider"))
    # session = await _ensure_session(state)

    provider = state.get("report_provider", "openai")
    model = state.get("report_model")

    args: Dict[str, Any] = {
        "job_id": state["job_id"],
        "structured": state["structured"],
        "qc": state["qc"],
        "match": None,
        "resume_text": state.get("resume_text"),
        "provider": provider,
        "temperature": 0.2,
        "max_tokens": 900,
    }
    if model:
        args["model"] = model
    # if provider == "nvidia": args["thinking"] = "disabled"  # only if supported

    res = await _call_tool_once("generate_report_api", args)
    payload = _tool_result_json(res, "generate_report_api")

    state["report_md"] = payload.get("report_md")
    state["report_meta"] = {k: payload.get(k) for k in ["usage", "meta"]}
    _trace(state, "generate_report_api", "ok", tokens=state["report_meta"].get("usage", {}).get("total_tokens"))
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
        return "report"

    extractor = (state.get("extract_meta") or {}).get("extractor") or {}
    # If we already used API, stop; otherwise fallback to API
    if extractor.get("provider") in ("openai", "nvidia"):
        return "finalize"
    return "extract_api"



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
