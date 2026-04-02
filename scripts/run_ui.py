from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/ui/app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
