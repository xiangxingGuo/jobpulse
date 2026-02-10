from pathlib import Path
from typing import Optional, Tuple

from src.db import get_conn, init_db
from src.text_clean.jd_clean import clean_jd


def pick_one_job(conn) -> Tuple[str, str]:
    """
    Pick the most recently scraped job that has a non-empty description.
    """
    row = conn.execute(
        """
        SELECT job_id, description
        FROM jobs
        WHERE description IS NOT NULL AND trim(description) != ''
        ORDER BY scraped_at_utc DESC
        LIMIT 1
        """
    ).fetchone()

    if not row:
        raise RuntimeError("No job found with non-empty description in jobs table.")

    return str(row[0]), str(row[1])


def export(job_id: str, raw_text: str) -> None:
    raw_dir = Path("data/raw/jd_raw")
    clean_dir = Path("data/raw/jd_txt")
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{job_id}.txt"
    clean_path = clean_dir / f"{job_id}.txt"

    raw_path.write_text(raw_text.strip() + "\n", encoding="utf-8")

    cleaned = clean_jd(raw_text)
    clean_path.write_text(cleaned.strip() + "\n", encoding="utf-8")

    print(f"Saved RAW JD to:   {raw_path}")
    print(f"Saved CLEAN JD to: {clean_path}")
    print("\nPreview (first 40 lines of CLEAN):")
    # for i, line in enumerate(cleaned.splitlines()[:40], start=1):
    #     print(f"{i:02d} | {line}")


def main() -> None:
    conn = get_conn()
    init_db(conn)

    # check raw path and clean path, find already exported job ids, and skip those
    raw_dir = Path("data/raw/jd_raw")
    clean_dir = Path("data/raw/jd_txt")
    finished_job_ids = set()
    if raw_dir.exists():
        finished_job_ids.update(p.stem for p in raw_dir.glob("*.txt"))
    if clean_dir.exists():
        finished_job_ids.update(p.stem for p in clean_dir.glob("*.txt"))
    
    rows = conn.execute(
        """
        SELECT job_id, description
        FROM jobs
        WHERE description IS NOT NULL AND trim(description) != ''
        ORDER BY scraped_at_utc DESC
        """
    ).fetchall()

    print(f"Found {len(rows)} jobs with non-empty descriptions in the database.")
    print(f"Found {len(finished_job_ids)} already exported job IDs in raw and clean directories.")

    for row in rows:
        job_id = str(row[0])
        raw_text = str(row[1])
        if job_id in finished_job_ids:
            print(f"Skipping already exported job_id: {job_id}")
            continue
        export(job_id, raw_text)


if __name__ == "__main__":
    main()
