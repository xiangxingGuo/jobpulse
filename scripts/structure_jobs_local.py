import argparse
import json
from typing import Optional

from src.db import get_conn, init_db, upsert_job_structured
from src.extractors.local_hf import LocalHFExtractor, SCHEMA_VERSION, PROMPT_VERSION

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # 3B：for 6GB-friendly 4-bit inference

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--only-missing", action="store_true", help="Only process jobs not yet structured")
    args = ap.parse_args()

    conn = get_conn()
    init_db(conn)

    if args.only_missing:
        rows = conn.execute("""
            SELECT j.job_id, j.title, j.company, j.location_text, j.description, j.url
            FROM jobs j
            LEFT JOIN jobs_structured s ON s.job_id = j.job_id
            WHERE s.job_id IS NULL
            ORDER BY j.scraped_at_utc DESC
            LIMIT ?
        """, (args.limit,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT job_id, title, company, location_text, description, url
            FROM jobs
            ORDER BY scraped_at_utc DESC
            LIMIT ?
        """, (args.limit,)).fetchall()

    print(f"Structuring {len(rows)} jobs using local model: {args.model}")

    extractor = LocalHFExtractor(model_name=args.model)

    ok = 0
    for i, (job_id, title, company, location_text, description, url) in enumerate(rows, 1):
        title = title or ""
        company = company or ""
        location_text = location_text or ""
        description = description or ""

        # Avoid huge prompts (keep it lean). You can tune this later.
        description_cut = description[:6000]

        debug = (i == 1)

        data, err = extractor.extract_with_retries(
            title=title,
            company=company,
            location=location_text,
            description=description_cut,
            retries=2,
            debug=debug,
            debug_id=job_id,
        )

        if data is None:
            upsert_job_structured(conn, {
                "job_id": job_id,
                "model": args.model,
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "data_json": json.dumps({"error": "extraction_failed"}, ensure_ascii=False),
                "confidence": 0.0,
                "error": err,
            })
            print(f"[{i}/{len(rows)}] ❌ {job_id} failed: {err}")
            continue

        conf = float(data.get("confidence", 0.5))
        upsert_job_structured(conn, {
            "job_id": job_id,
            "model": args.model,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "data_json": json.dumps(data, ensure_ascii=False),
            "confidence": conf,
            "error": None,
        })
        ok += 1
        print(f"[{i}/{len(rows)}] ✅ {job_id} {data.get('role_category')} {data.get('work_mode')} conf={conf:.2f}")

    print(f"Done. Success: {ok}/{len(rows)}")

if __name__ == "__main__":
    main()
