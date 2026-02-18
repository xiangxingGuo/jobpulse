from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ARTIFACTS_DIR = Path("data/artifacts")


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _tool_result_text(res: Any) -> str:
    # res.content is a list of content blocks; FastMCP typically returns one TextContent
    parts = []
    for b in getattr(res, "content", []) or []:
        t = getattr(b, "text", None)
        if isinstance(t, str):
            parts.append(t)
    return "".join(parts)

def _tool_result_json(res: Any, tool_name: str) -> Any:
    # If tool failed, surface the error text
    if getattr(res, "is_error", False):
        msg = _tool_result_text(res).strip()
        raise RuntimeError(f"Tool {tool_name} failed: {msg or res}")

    txt = _tool_result_text(res).strip()
    if not txt:
        raise RuntimeError(f"Tool {tool_name} returned empty content: {res}")

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # show a snippet to debug
        snippet = txt[:500]
        raise RuntimeError(f"Tool {tool_name} returned non-JSON text (first 500 chars): {snippet}")



async def run_one(job_id: str, provider: str, model: str | None, prompt_name: str, out_dir: Path) -> None:
    # Spawn MCP server as a stdio subprocess and connect to it
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp_server.server"],
        # Ensure env is passed through (important for OPENAI_API_KEY / NVIDIA_API_KEY)
        env=dict(os.environ),
    )

    trace: list[dict[str, Any]] = []
    t0 = time.time()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            trace.append({"t": _ts(), "step": "mcp.initialize", "status": "start"})
            await session.initialize()
            trace.append({"t": _ts(), "step": "mcp.initialize", "status": "ok"})

            job_dir = out_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            # 1) fetch_jd
            trace.append({"t": _ts(), "step": "fetch_jd", "status": "start"})
            fetch_res = await session.call_tool("fetch_jd", {"job_id": job_id, "source": "handshake"})
            # MCP returns ToolResult; its content is usually a list of typed blocks.
            # For FastMCP, the JSON result is typically in content[0].text
            fetch_payload = _tool_result_json(fetch_res, "fetch_jd")
            trace.append({"t": _ts(), "step": "fetch_jd", "status": "ok", "len": len(fetch_payload.get("jd_text", ""))})

            jd_text = fetch_payload["jd_text"]
            _write_text(job_dir / "jd.txt", jd_text)
            _write_json(job_dir / "fetch.json", fetch_payload)

            # 2) extract_api (OpenAI-compatible; provider can be openai/nvidia)
            trace.append({"t": _ts(), "step": "extract_api", "status": "start", "provider": provider})
            extract_args: Dict[str, Any] = {
                "job_id": job_id,
                "jd_text": jd_text,
                "prompt_name": prompt_name,
                "provider": provider,
                "temperature": 0.0,
                "max_tokens": 1200,
            }
            if model:
                extract_args["model"] = model
            # If you added thinking control earlier, you can pass:
            # if provider == "nvidia": extract_args["thinking"] = "disabled"

            ext_res = await session.call_tool("extract_api", extract_args)
            ext_payload = _tool_result_json(ext_res, "extract_api")
            trace.append({"t": _ts(), "step": "extract_api", "status": "ok", "parse_ok": ext_payload.get("parse_ok")})

            _write_json(job_dir / "structured.json", ext_payload.get("structured"))
            _write_json(job_dir / "extract_meta.json", {k: ext_payload.get(k) for k in ["parse_ok", "parse_repaired", "usage", "extractor"]})
            _write_text(job_dir / "extract_raw.txt", ext_payload.get("raw_output", ""))

            # 3) qc_validate
            trace.append({"t": _ts(), "step": "qc_validate", "status": "start"})
            qc_args = {
                "job_id": job_id,
                "structured": ext_payload.get("structured"),
                "parse_ok": ext_payload.get("parse_ok", False),
                "parse_repaired": ext_payload.get("parse_repaired", False),
                "extractor": ext_payload.get("extractor", {}),
                "require_keys": ["role_title", "company", "requirements", "responsibilities"],
                "require_non_empty_any_of": [["requirements", "responsibilities"]],
            }
            qc_res = await session.call_tool("qc_validate", qc_args)
            qc_payload = _tool_result_json(qc_res, "qc_validate")
            trace.append({"t": _ts(), "step": "qc_validate", "status": "ok", "qc": qc_payload.get("status")})
            _write_json(job_dir / "qc.json", qc_payload)

            # 4) generate_report_api (only if QC pass; otherwise still generate with warning if you want)
            if qc_payload.get("status") == "pass" and ext_payload.get("structured"):
                trace.append({"t": _ts(), "step": "generate_report_api", "status": "start", "provider": provider})
                rep_args: Dict[str, Any] = {
                    "job_id": job_id,
                    "structured": ext_payload["structured"],
                    "qc": qc_payload,
                    "match": None,
                    "resume_text": None,
                    "provider": provider,
                    "temperature": 0.2,
                    "max_tokens": 900,
                }
                if model:
                    rep_args["model"] = model

                rep_res = await session.call_tool("generate_report_api", rep_args)
                rep_payload = _tool_result_json(rep_res, "generate_report_api")
                trace.append({"t": _ts(), "step": "generate_report_api", "status": "ok", "tokens": rep_payload.get("usage", {}).get("total_tokens")})

                _write_text(job_dir / "report.md", rep_payload.get("report_md", ""))
                _write_json(job_dir / "report_meta.json", {k: rep_payload.get(k) for k in ["usage", "meta"]})
            else:
                trace.append({"t": _ts(), "step": "generate_report_api", "status": "skipped", "reason": "qc_fail_or_no_structured"})

    trace.append({"t": _ts(), "step": "done", "status": "ok", "elapsed_s": round(time.time() - t0, 2)})
    _write_json(out_dir / job_id / "trace.json", trace)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--provider", default="openai", choices=["openai", "nvidia"])
    ap.add_argument("--model", default=None, help="override model id (optional)")
    ap.add_argument("--prompt-name", default="jd_extract_v2")
    ap.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    args = ap.parse_args()

    asyncio.run(run_one(args.job_id, args.provider, args.model, args.prompt_name, Path(args.out_dir)))


if __name__ == "__main__":
    main()
