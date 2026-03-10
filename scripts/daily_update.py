from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


def main() -> None:
    run([sys.executable, "scripts/run_pipeline.py", "--pages", "1", "--limit", "20"])
    run([sys.executable, "scripts/build_vector_index.py"])


if __name__ == "__main__":
    main()