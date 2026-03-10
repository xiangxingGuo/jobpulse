from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.db import fetch_jobs_for_retrieval
from src.retrieval.documents import build_documents
from src.retrieval.embed import EmbeddingModel
from src.retrieval.faiss_index import JobFaissIndex

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="data/vectors")
    args = ap.parse_args()

    rows = fetch_jobs_for_retrieval(limit=args.limit)
    docs = build_documents(rows)

    texts = [d.text for d in docs]
    meta = [d.metadata for d in docs]

    model = EmbeddingModel(args.model)
    vecs = model.encode(texts)

    index = JobFaissIndex(vecs.shape[1])
    index.add(vecs, meta)
    index.save(args.out_dir)

    summary = {
        "count": len(docs),
        "dim": int(vecs.shape[1]),
        "model": args.model,
        "out_dir": args.out_dir,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()