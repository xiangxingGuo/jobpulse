from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from src.orch.graph import build_graph

ARTIFACTS_DIR = Path("data/artifacts")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    app = build_graph()

    state: Dict[str, Any] = {
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
    }

    out = await app.ainvoke(state)

    job_dir = Path(args.out_dir) / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    _write_text(job_dir / "jd.txt", out.get("jd_text", ""))
    _write_json(job_dir / "structured.json", out.get("structured"))
    _write_json(job_dir / "extract_meta.json", out.get("extract_meta", {}))
    _write_json(job_dir / "qc.json", out.get("qc", {}))
    _write_text(job_dir / "report.md", out.get("report_md", ""))
    _write_json(job_dir / "report_meta.json", out.get("report_meta", {}))
    _write_json(job_dir / "trace.json", out.get("trace", []))

    print(f"✅ Wrote artifacts to: {job_dir}")


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
