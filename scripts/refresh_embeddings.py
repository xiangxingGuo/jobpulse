from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.db import (
    init_db,
    fetch_jobs_needing_reindex,
    upsert_job_embedding_record,
)
from src.retrieval.documents import build_documents
from src.retrieval.embed import EmbeddingModel
from src.retrieval.faiss_index import JobFaissIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="data/vectors")
    args = ap.parse_args()

    init_db()

    rows = fetch_jobs_needing_reindex(embedding_model=args.model, limit=args.limit)
    docs = build_documents(rows)

    if not docs:
        summary = {
            "status": "noop",
            "jobs_reindexed": 0,
            "model": args.model,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    texts = [d.text for d in docs]
    meta = [d.metadata for d in docs]

    model = EmbeddingModel(args.model)
    vecs = model.encode(texts)

    # simple first version: rebuild full index after selecting changed jobs is harder to merge cleanly.
    # for now, re-run build_vector_index after updating records, or replace this later with full upsertable storage.
    # easiest practical version: just call your full builder after marking what changed.

    index = JobFaissIndex(vecs.shape[1])
    index.add(vecs, meta)
    index.save(args.out_dir)

    for row in rows:
        upsert_job_embedding_record(
            job_id=str(row["job_id"]),
            content_hash=row.get("content_hash"),
            embedding_model=args.model,
        )

    summary = {
        "status": "ok",
        "jobs_reindexed": len(rows),
        "model": args.model,
        "out_dir": args.out_dir,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, "refresh_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()