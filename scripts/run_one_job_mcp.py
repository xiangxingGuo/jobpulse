from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ARTIFACTS_DIR = Path(os.getenv("ARTIFACT_DIR", "data/artifacts")) / "mcp"


# ----------------------------
# IO helpers
# ----------------------------


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _tool_result_text(res: Any) -> str:
    parts = []
    for b in getattr(res, "content", []) or []:
        t = getattr(b, "text", None)
        if isinstance(t, str):
            parts.append(t)
    return "".join(parts)


def _tool_result_json(res: Any, tool_name: str) -> Any:
    if getattr(res, "is_error", False):
        msg = _tool_result_text(res).strip()
        raise RuntimeError(f"Tool {tool_name} failed: {msg or res}")

    txt = _tool_result_text(res).strip()
    if not txt:
        raise RuntimeError(f"Tool {tool_name} returned empty content: {res}")

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        snippet = txt[:500]
        raise RuntimeError(f"Tool {tool_name} returned non-JSON text (first 500 chars): {snippet}")


# ----------------------------
# Config & Metrics
# ----------------------------


@dataclass(frozen=True)
class RunOneConfig:
    job_id: str
    source: str = "handshake"

    # extraction routing
    local_first: bool = True
    fallback_to_api: bool = True

    # api provider/model
    provider: str = "openai"  # openai|nvidia
    model: Optional[str] = None
    prompt_name: str = "jd_extract_v2"

    # generation parameters
    temperature_extract: float = 0.0
    max_tokens_extract: int = 1200
    temperature_report: float = 0.2
    max_tokens_report: int = 900

    # QC policy
    require_keys: Tuple[str, ...] = ("role_title", "company", "requirements", "responsibilities")
    require_non_empty_any_of: Tuple[Tuple[str, ...], ...] = (("requirements", "responsibilities"),)

    # artifacts
    out_dir: Path = ARTIFACTS_DIR


class RunOneMetrics:
    def __init__(self) -> None:
        self.step_ms: Dict[str, int] = {}
        self.route: str = "unknown"
        self.fallback_reason: Optional[str] = None
        self.local = {"attempted": False, "parse_ok": None, "qc": None}
        self.api = {"attempted": False, "parse_ok": None, "qc": None}
        self.final_qc: Optional[str] = None

    def record_ms(self, step: str, dt_s: float) -> None:
        self.step_ms[step] = int(dt_s * 1000)

    def summary(self, elapsed_s: float) -> Dict[str, Any]:
        return {
            "elapsed_s": round(elapsed_s, 3),
            "route": self.route,
            "fallback_reason": self.fallback_reason,
            "steps_ms": self.step_ms,
            "local": self.local,
            "api": self.api,
            "final_qc": self.final_qc,
            "slo": {
                "final_qc_pass": (self.final_qc == "pass"),
            },
        }


# ----------------------------
# Core runner
# ----------------------------


async def _call_tool(
    session: ClientSession, trace: list[dict[str, Any]], name: str, args: Dict[str, Any]
) -> Any:
    t0 = time.time()
    trace.append({"t": _ts(), "step": name, "status": "start", "args_keys": sorted(args.keys())})
    res = await session.call_tool(name, args)
    payload = _tool_result_json(res, name)
    dt = time.time() - t0
    trace.append({"t": _ts(), "step": name, "status": "ok", "elapsed_s": round(dt, 3)})
    return payload, dt


async def run_one(cfg: RunOneConfig) -> None:
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp_server.server"],
        env=dict(os.environ),
    )

    trace: list[dict[str, Any]] = []
    metrics = RunOneMetrics()
    t0 = time.time()

    job_dir = cfg.out_dir / cfg.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(job_dir / "run_one_config.json", asdict(cfg) | {"out_dir": str(cfg.out_dir)})

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            trace.append({"t": _ts(), "step": "mcp.initialize", "status": "start"})
            await session.initialize()
            trace.append({"t": _ts(), "step": "mcp.initialize", "status": "ok"})

            # 1) fetch_jd
            fetch_payload, dt = await _call_tool(
                session, trace, "fetch_jd", {"job_id": cfg.job_id, "source": cfg.source}
            )
            metrics.record_ms("fetch_jd", dt)
            jd_text = fetch_payload.get("jd_text", "")
            _write_text(job_dir / "jd.txt", jd_text)
            _write_json(job_dir / "fetch.json", fetch_payload)

            # Helper to run QC
            async def qc(
                structured: Any, parse_ok: bool, parse_repaired: bool, extractor: dict
            ) -> Tuple[dict, float]:
                qc_args = {
                    "job_id": cfg.job_id,
                    "structured": structured,
                    "parse_ok": parse_ok,
                    "parse_repaired": parse_repaired,
                    "extractor": extractor,
                    "require_keys": list(cfg.require_keys),
                    "require_non_empty_any_of": [list(x) for x in cfg.require_non_empty_any_of],
                }
                return await _call_tool(session, trace, "qc_validate", qc_args)

            # 2) try local extraction (optional)
            local_ok = False
            local_payload = None
            qc_local = None

            if cfg.local_first:
                metrics.local["attempted"] = True
                try:
                    local_payload, dt = await _call_tool(
                        session,
                        trace,
                        "extract_local",
                        {
                            "job_id": cfg.job_id,
                            "jd_text": jd_text,
                            "prompt_name": cfg.prompt_name,
                            "temperature": cfg.temperature_extract,
                            "max_tokens": cfg.max_tokens_extract,
                        },
                    )
                    metrics.record_ms("extract_local", dt)

                    # normalize expected keys similar to extract_api tool
                    structured = local_payload.get("structured")
                    parse_ok = bool(local_payload.get("parse_ok", False))
                    parse_repaired = bool(local_payload.get("parse_repaired", False))
                    extractor = local_payload.get("extractor", {"mode": "local"})

                    metrics.local["parse_ok"] = parse_ok
                    _write_json(job_dir / "structured_local.json", structured)
                    _write_json(
                        job_dir / "extract_local_meta.json",
                        {
                            k: local_payload.get(k)
                            for k in ["parse_ok", "parse_repaired", "usage", "extractor"]
                        },
                    )
                    _write_text(
                        job_dir / "extract_local_raw.txt", local_payload.get("raw_output", "")
                    )

                    qc_local, dt = await qc(structured, parse_ok, parse_repaired, extractor)
                    metrics.record_ms("qc_local", dt)
                    metrics.local["qc"] = qc_local.get("status")

                    _write_json(job_dir / "qc_local.json", qc_local)

                    local_ok = (qc_local.get("status") == "pass") and structured is not None
                except Exception as e:
                    trace.append(
                        {
                            "t": _ts(),
                            "step": "extract_local",
                            "status": "fail",
                            "error": str(e)[:400],
                        }
                    )
                    metrics.local["parse_ok"] = False
                    metrics.local["qc"] = "error"
                    metrics.fallback_reason = metrics.fallback_reason or "local_error"

            # 3) if local ok -> report; else fallback to api if enabled
            chosen_structured = None
            # chosen_extractor = None
            chosen_qc = None

            if local_ok:
                metrics.route = "local_only"
                chosen_structured = local_payload.get("structured")
                # chosen_extractor = local_payload.get("extractor", {"mode": "local"})
                chosen_qc = qc_local
            else:
                if cfg.local_first and cfg.fallback_to_api:
                    metrics.route = "local_then_api"
                    if metrics.fallback_reason is None:
                        # qc fail or parse fail are the common causes
                        if qc_local and qc_local.get("status") != "pass":
                            metrics.fallback_reason = "qc_fail_local"
                        else:
                            metrics.fallback_reason = "local_not_usable"
                else:
                    metrics.route = "api_only"

                if not cfg.fallback_to_api:
                    metrics.final_qc = qc_local.get("status") if qc_local else "fail"
                    _write_json(job_dir / "trace.json", trace)
                    _write_json(job_dir / "run_one_summary.json", metrics.summary(time.time() - t0))
                    return

                # ---- extract_api
                metrics.api["attempted"] = True
                extract_args: Dict[str, Any] = {
                    "job_id": cfg.job_id,
                    "jd_text": jd_text,
                    "prompt_name": cfg.prompt_name,
                    "provider": cfg.provider,
                    "temperature": cfg.temperature_extract,
                    "max_tokens": cfg.max_tokens_extract,
                }
                if cfg.model:
                    extract_args["model"] = cfg.model

                api_payload, dt = await _call_tool(session, trace, "extract_api", extract_args)
                metrics.record_ms("extract_api", dt)

                structured = api_payload.get("structured")
                parse_ok = bool(api_payload.get("parse_ok", False))
                parse_repaired = bool(api_payload.get("parse_repaired", False))
                extractor = api_payload.get("extractor", {"mode": "api", "provider": cfg.provider})

                metrics.api["parse_ok"] = parse_ok
                _write_json(job_dir / "structured_api.json", structured)
                _write_json(
                    job_dir / "extract_api_meta.json",
                    {
                        k: api_payload.get(k)
                        for k in ["parse_ok", "parse_repaired", "usage", "extractor"]
                    },
                )
                _write_text(job_dir / "extract_api_raw.txt", api_payload.get("raw_output", ""))

                qc_api, dt = await qc(structured, parse_ok, parse_repaired, extractor)
                metrics.record_ms("qc_api", dt)
                metrics.api["qc"] = qc_api.get("status")
                _write_json(job_dir / "qc_api.json", qc_api)

                chosen_structured = structured
                # chosen_extractor = extractor
                chosen_qc = qc_api

            # 4) generate_report_api (only if QC pass)
            if chosen_qc and chosen_qc.get("status") == "pass" and chosen_structured:
                rep_args: Dict[str, Any] = {
                    "job_id": cfg.job_id,
                    "structured": chosen_structured,
                    "qc": chosen_qc,
                    "match": None,
                    "resume_text": None,
                    "provider": cfg.provider,
                    "temperature": cfg.temperature_report,
                    "max_tokens": cfg.max_tokens_report,
                }
                if cfg.model:
                    rep_args["model"] = cfg.model

                rep_payload, dt = await _call_tool(session, trace, "generate_report_api", rep_args)
                metrics.record_ms("generate_report_api", dt)

                _write_text(job_dir / "report.md", rep_payload.get("report_md", ""))
                _write_json(
                    job_dir / "report_meta.json", {k: rep_payload.get(k) for k in ["usage", "meta"]}
                )
            else:
                trace.append(
                    {
                        "t": _ts(),
                        "step": "generate_report_api",
                        "status": "skipped",
                        "reason": "qc_fail_or_no_structured",
                    }
                )

            metrics.final_qc = chosen_qc.get("status") if chosen_qc else "fail"
            elapsed = time.time() - t0
            trace.append(
                {"t": _ts(), "step": "done", "status": "ok", "elapsed_s": round(elapsed, 2)}
            )

            _write_json(job_dir / "trace.json", trace)
            _write_json(job_dir / "run_one_summary.json", metrics.summary(elapsed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--provider", default="openai", choices=["openai", "nvidia"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt-name", default="jd_extract_v2")
    ap.add_argument("--out-dir", default=str(ARTIFACTS_DIR))

    ap.add_argument("--local-first", action="store_true", default=True)
    ap.add_argument("--no-local-first", dest="local_first", action="store_false")
    ap.add_argument("--fallback-to-api", action="store_true", default=True)
    ap.add_argument("--no-fallback-to-api", dest="fallback_to_api", action="store_false")

    args = ap.parse_args()

    cfg = RunOneConfig(
        job_id=args.job_id,
        provider=args.provider,
        model=args.model,
        prompt_name=args.prompt_name,
        out_dir=Path(args.out_dir),
        local_first=args.local_first,
        fallback_to_api=args.fallback_to_api,
    )

    asyncio.run(run_one(cfg))


if __name__ == "__main__":
    main()
