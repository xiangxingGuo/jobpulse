from __future__ import annotations

import argparse
import json

from src.retrieval.search import JobSearchService


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--job-id", type=str, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    svc = JobSearchService()

    if args.query:
        results = svc.search_jobs(args.query, top_k=args.top_k)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if args.job_id:
        results = svc.similar_jobs(args.job_id, top_k=args.top_k)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    raise SystemExit("Provide either --query or --job-id")


if __name__ == "__main__":
    main()
