from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="1")
    ap.add_argument("--limit", default="20")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    t0 = time.time()

    print(f"[START] daily update at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    run(
        [
            sys.executable,
            "scripts/run_pipeline.py",
            "--pages",
            str(args.pages),
            "--limit",
            str(args.limit),
        ],
        cwd=project_root,
    )

    run(
        [
            sys.executable,
            "scripts/build_vector_index.py",
            "--model",
            args.model,
        ],
        cwd=project_root,
    )

    print(
        f"[DONE] daily update finished in {round(time.time() - t0, 2)} sec",
        flush=True,
    )

if __name__ == "__main__":
    main()