from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

DB_PATH = Path("data/db/jobs.db")

BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  title TEXT,
  company TEXT,
  posted_text TEXT,
  apply_by_text TEXT,
  pay_text TEXT,
  location_text TEXT,
  employment_type TEXT,
  date_range_text TEXT,
  work_auth_text TEXT,
  opt_cpt_text TEXT,
  description TEXT,
  scraped_at_utc TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS job_skills (
  job_id TEXT NOT NULL,
  skill TEXT NOT NULL,
  PRIMARY KEY (job_id, skill),
  FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS jobs_structured (
  job_id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  data_json TEXT NOT NULL,
  confidence REAL,
  error TEXT,
  extracted_at_utc TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_structured_model ON jobs_structured(model);
"""

AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS scrape_runs (
  run_id TEXT PRIMARY KEY,
  started_at_utc TEXT NOT NULL,
  finished_at_utc TEXT,
  config_json TEXT,
  summary_json TEXT,
  elapsed_sec REAL,
  slo_met INTEGER
);

CREATE TABLE IF NOT EXISTS scrape_events (
  run_id TEXT NOT NULL,
  job_id TEXT,
  url TEXT,
  stage TEXT NOT NULL,            -- collect_links, parse_detail, gate, db_upsert, extract_skills...
  status TEXT NOT NULL,           -- ok, fail, gated
  reason TEXT,                    -- gate_desc_too_short, parse_timeout, db_error, etc.
  elapsed_ms INTEGER,
  details_json TEXT,
  ts_utc TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_scrape_events_run ON scrape_events(run_id);
CREATE INDEX IF NOT EXISTS idx_scrape_events_job ON scrape_events(job_id);
"""

EMBED_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_embeddings (
  job_id TEXT PRIMARY KEY,
  content_hash TEXT,
  embedding_model TEXT NOT NULL,
  indexed_at_utc TEXT NOT NULL,
  FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_job_embeddings_model ON job_embeddings(embedding_model);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _table_info(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    # row[1] is name
    return [r[1] for r in rows]


def _add_column_if_missing(conn: sqlite3.Connection, table: str, coldef: str, colname: str) -> None:
    cols = set(_table_info(conn, table))
    if colname not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef};")


def migrate(conn: sqlite3.Connection) -> None:
    """
    Add industrial-grade columns & audit tables without breaking existing DB.
    """
    # jobs: add fields for reliability / idempotency / observability
    _add_column_if_missing(conn, "jobs", "scrape_status TEXT", "scrape_status")
    _add_column_if_missing(conn, "jobs", "scrape_error TEXT", "scrape_error")
    _add_column_if_missing(conn, "jobs", "content_hash TEXT", "content_hash")
    _add_column_if_missing(conn, "jobs", "last_seen_at_utc TEXT", "last_seen_at_utc")


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(BASE_SCHEMA)
        conn.executescript(AUDIT_SCHEMA)
        conn.executescript(EMBED_SCHEMA)
        migrate(conn)
        conn.commit()


def get_existing_job_ids() -> Set[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT job_id FROM jobs;").fetchall()
        return {r[0] for r in rows if r and r[0]}


def begin_run(run_id: str, config: Dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scrape_runs (run_id, started_at_utc, config_json) VALUES (?, ?, ?)",
            (run_id, _utc_now_iso(), json.dumps(config, ensure_ascii=False)),
        )
        conn.commit()


def end_run(
    run_id: str, summary: Dict[str, Any], elapsed_sec: float, slo_met: Optional[bool]
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE scrape_runs
            SET finished_at_utc=?, summary_json=?, elapsed_sec=?, slo_met=?
            WHERE run_id=?
            """,
            (
                _utc_now_iso(),
                json.dumps(summary, ensure_ascii=False),
                float(elapsed_sec),
                None if slo_met is None else (1 if slo_met else 0),
                run_id,
            ),
        )
        conn.commit()


def record_event(
    run_id: str,
    stage: str,
    status: str,
    job_id: Optional[str] = None,
    url: Optional[str] = None,
    reason: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO scrape_events (run_id, job_id, url, stage, status, reason, elapsed_ms, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                job_id,
                url,
                stage,
                status,
                reason,
                elapsed_ms,
                None if details is None else json.dumps(details, ensure_ascii=False),
            ),
        )
        conn.commit()


def upsert_job(
    row: Dict[str, Any], scrape_status: str = "success", scrape_error: Optional[str] = None
) -> None:
    cols = [
        "job_id",
        "url",
        "title",
        "company",
        "posted_text",
        "apply_by_text",
        "pay_text",
        "location_text",
        "employment_type",
        "date_range_text",
        "work_auth_text",
        "opt_cpt_text",
        "description",
        "scrape_status",
        "scrape_error",
        "content_hash",
        "last_seen_at_utc",
    ]
    data = {c: row.get(c) for c in cols}

    # Set operational fields
    data["scrape_status"] = scrape_status
    data["scrape_error"] = scrape_error
    data["last_seen_at_utc"] = data.get("last_seen_at_utc") or _utc_now_iso()

    # content_hash for idempotency
    desc = row.get("description") or ""
    data["content_hash"] = row.get("content_hash") or (
        None
        if not desc
        else __import__("hashlib").sha1(desc.encode("utf-8", errors="ignore")).hexdigest()
    )

    placeholders = ",".join(["?"] * len(cols))
    assignments = ",".join([f"{c}=excluded.{c}" for c in cols[1:]])  # don't overwrite PK
    sql = f"""
    INSERT INTO jobs ({",".join(cols)}) VALUES ({placeholders})
    ON CONFLICT(job_id) DO UPDATE SET {assignments},
      scraped_at_utc = datetime('now')
    """

    with get_conn() as conn:
        conn.execute(sql, [data[c] for c in cols])
        conn.commit()


def replace_job_skills(job_id: str, skills: Iterable[str]) -> None:
    uniq = sorted(set([s.strip() for s in skills if s and s.strip()]))
    with get_conn() as conn:
        conn.execute("DELETE FROM job_skills WHERE job_id=?", (job_id,))
        conn.executemany(
            "INSERT OR IGNORE INTO job_skills (job_id, skill) VALUES (?, ?)",
            [(job_id, s) for s in uniq],
        )
        conn.commit()


def upsert_job_structured(row: Dict[str, Any]) -> None:
    cols = [
        "job_id",
        "model",
        "schema_version",
        "prompt_version",
        "data_json",
        "confidence",
        "error",
    ]
    placeholders = ",".join(["?"] * len(cols))
    assignments = ",".join([f"{c}=excluded.{c}" for c in cols[1:]])
    sql = f"""
    INSERT INTO jobs_structured ({",".join(cols)}) VALUES ({placeholders})
    ON CONFLICT(job_id) DO UPDATE SET {assignments},
      extracted_at_utc = datetime('now')
    """
    with get_conn() as conn:
        conn.execute(sql, [row.get(c) for c in cols])
        conn.commit()


def get_job_content_hash(job_id: str) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT content_hash FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return row[0] if row and row[0] else None


def update_job_operational(job_id: str, scrape_status: str, scrape_error: Optional[str]) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET scrape_status=?, scrape_error=?, scraped_at_utc=datetime('now')
            WHERE job_id=?
            """,
            (scrape_status, scrape_error, job_id),
        )
        conn.commit()


def fetch_jobs_for_retrieval(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch jobs with skills and structured extraction for embedding / retrieval.
    """
    with get_conn() as conn:
        sql = """
        SELECT
            j.job_id,
            j.url,
            j.title,
            j.company,
            j.location_text,
            j.description,
            j.content_hash,
            js.data_json
        FROM jobs j
        LEFT JOIN jobs_structured js
          ON j.job_id = js.job_id
        """

        if limit:
            sql += " LIMIT ?"
            rows = conn.execute(sql, (limit,)).fetchall()
        else:
            rows = conn.execute(sql).fetchall()

        jobs = []

        for r in rows:
            (
                job_id,
                url,
                title,
                company,
                location_text,
                description,
                content_hash,
                structured_json,
            ) = r

            # load structured extraction
            structured = None
            if structured_json:
                try:
                    structured = json.loads(structured_json)
                except Exception:
                    structured = None

            # fetch skills
            skills_rows = conn.execute(
                "SELECT skill FROM job_skills WHERE job_id=?",
                (job_id,),
            ).fetchall()

            skills = [s[0] for s in skills_rows]

            jobs.append(
                {
                    "job_id": job_id,
                    "url": url,
                    "title": title,
                    "company": company,
                    "location_text": location_text,
                    "description": description,
                    "skills": skills,
                    "structured": structured,
                    "content_hash": content_hash,
                }
            )

        return jobs


def fetch_job_detail(job_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                j.job_id,
                j.url,
                j.title,
                j.company,
                j.location_text,
                j.description,
                j.scrape_status,
                j.scrape_error,
                j.content_hash,
                j.last_seen_at_utc,
                js.model,
                js.schema_version,
                js.prompt_version,
                js.data_json,
                js.confidence,
                js.error,
                js.extracted_at_utc
            FROM jobs j
            LEFT JOIN jobs_structured js
              ON j.job_id = js.job_id
            WHERE j.job_id = ?
            """,
            (job_id,),
        ).fetchone()

        if not row:
            return None

        (
            job_id,
            url,
            title,
            company,
            location_text,
            description,
            scrape_status,
            scrape_error,
            content_hash,
            last_seen_at_utc,
            model,
            schema_version,
            prompt_version,
            data_json,
            confidence,
            structured_error,
            extracted_at_utc,
        ) = row

        skills_rows = conn.execute(
            "SELECT skill FROM job_skills WHERE job_id=? ORDER BY skill ASC",
            (job_id,),
        ).fetchall()
        skills = [r[0] for r in skills_rows]

        structured = None
        if data_json:
            try:
                structured = json.loads(data_json)
            except Exception:
                structured = None

        return {
            "job_id": job_id,
            "url": url,
            "title": title,
            "company": company,
            "location_text": location_text,
            "description": description,
            "skills": skills,
            "scrape_status": scrape_status,
            "scrape_error": scrape_error,
            "content_hash": content_hash,
            "last_seen_at_utc": last_seen_at_utc,
            "structured": structured,
            "structured_meta": {
                "model": model,
                "schema_version": schema_version,
                "prompt_version": prompt_version,
                "confidence": confidence,
                "error": structured_error,
                "extracted_at_utc": extracted_at_utc,
            },
        }


def fetch_recent_scrape_runs(limit: int = 10) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                run_id,
                started_at_utc,
                finished_at_utc,
                summary_json,
                elapsed_sec,
                slo_met
            FROM scrape_runs
            ORDER BY started_at_utc DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            run_id, started_at_utc, finished_at_utc, summary_json, elapsed_sec, slo_met = r

            summary = None
            if summary_json:
                try:
                    summary = json.loads(summary_json)
                except Exception:
                    summary = None

            out.append(
                {
                    "run_id": run_id,
                    "started_at_utc": started_at_utc,
                    "finished_at_utc": finished_at_utc,
                    "elapsed_sec": elapsed_sec,
                    "slo_met": None if slo_met is None else bool(slo_met),
                    "summary": summary,
                }
            )
        return out


def fetch_metrics_summary(limit: int = 20) -> Dict[str, Any]:
    runs = fetch_recent_scrape_runs(limit=limit)

    if not runs:
        return {
            "runs_considered": 0,
            "scrape": {
                "avg_elapsed_sec": None,
                "slo_pass_rate": None,
                "success_rate_avg": None,
                "dq_slo_pass_rate": None,
            },
            "counts": {},
            "latest_runs": [],
        }

    elapsed_vals = []
    slo_flags = []
    success_rates = []
    dq_slo_flags = []

    agg_counts: Dict[str, int] = {}

    for run in runs:
        if run.get("elapsed_sec") is not None:
            elapsed_vals.append(float(run["elapsed_sec"]))

        if run.get("slo_met") is not None:
            slo_flags.append(bool(run["slo_met"]))

        summary = run.get("summary") or {}

        slo = summary.get("slo") or {}
        if slo.get("success_rate") is not None:
            success_rates.append(float(slo["success_rate"]))

        dq_slo = summary.get("dq_slo") or {}
        if dq_slo.get("met") is not None:
            dq_slo_flags.append(bool(dq_slo["met"]))

        counts = summary.get("counts") or {}
        for k, v in counts.items():
            try:
                agg_counts[k] = agg_counts.get(k, 0) + int(v)
            except Exception:
                pass

    def _avg(xs):
        return (sum(xs) / len(xs)) if xs else None

    def _rate(flags):
        return (sum(1 for x in flags if x) / len(flags)) if flags else None

    latest_runs = []
    for run in runs[:10]:
        summary = run.get("summary") or {}
        latest_runs.append(
            {
                "run_id": run.get("run_id"),
                "started_at_utc": run.get("started_at_utc"),
                "elapsed_sec": run.get("elapsed_sec"),
                "slo_met": run.get("slo_met"),
                "success_rate": ((summary.get("slo") or {}).get("success_rate")),
                "dq_slo_met": ((summary.get("dq_slo") or {}).get("met")),
                "jobs_upserted": ((summary.get("counts") or {}).get("jobs_upserted")),
                "jobs_parsed_ok": ((summary.get("counts") or {}).get("jobs_parsed_ok")),
            }
        )

    return {
        "runs_considered": len(runs),
        "scrape": {
            "avg_elapsed_sec": _avg(elapsed_vals),
            "slo_pass_rate": _rate(slo_flags),
            "success_rate_avg": _avg(success_rates),
            "dq_slo_pass_rate": _rate(dq_slo_flags),
        },
        "counts": agg_counts,
        "latest_runs": latest_runs,
    }


def upsert_job_embedding_record(
    job_id: str, content_hash: Optional[str], embedding_model: str
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO job_embeddings (job_id, content_hash, embedding_model, indexed_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
              content_hash=excluded.content_hash,
              embedding_model=excluded.embedding_model,
              indexed_at_utc=excluded.indexed_at_utc
            """,
            (job_id, content_hash, embedding_model, _utc_now_iso()),
        )
        conn.commit()


def get_embedding_record(job_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT job_id, content_hash, embedding_model, indexed_at_utc
            FROM job_embeddings
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "job_id": row[0],
            "content_hash": row[1],
            "embedding_model": row[2],
            "indexed_at_utc": row[3],
        }


def fetch_jobs_needing_reindex(
    embedding_model: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    rows = fetch_jobs_for_retrieval(limit=None)

    out: List[Dict[str, Any]] = []
    for row in rows:
        rec = get_embedding_record(str(row["job_id"]))
        if rec is None:
            out.append(row)
            continue
        if rec.get("embedding_model") != embedding_model:
            out.append(row)
            continue
        if rec.get("content_hash") != row.get("content_hash"):
            out.append(row)
            continue

    if limit is not None:
        return out[:limit]
    return out


def fetch_analytics_summary(limit: int = 10) -> Dict[str, Any]:
    with get_conn() as conn:
        total_jobs_row = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        total_jobs = int(total_jobs_row[0]) if total_jobs_row else 0

        top_skills_rows = conn.execute(
            """
            SELECT skill, COUNT(*) AS cnt
            FROM job_skills
            GROUP BY skill
            ORDER BY cnt DESC, skill ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        top_companies_rows = conn.execute(
            """
            SELECT company, COUNT(*) AS cnt
            FROM jobs
            WHERE company IS NOT NULL AND TRIM(company) <> ''
            GROUP BY company
            ORDER BY cnt DESC, company ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        top_locations_rows = conn.execute(
            """
            SELECT location_text, COUNT(*) AS cnt
            FROM jobs
            WHERE location_text IS NOT NULL AND TRIM(location_text) <> ''
            GROUP BY location_text
            ORDER BY cnt DESC, location_text ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        top_titles_rows = conn.execute(
            """
            SELECT title, COUNT(*) AS cnt
            FROM jobs
            WHERE title IS NOT NULL AND TRIM(title) <> ''
            GROUP BY title
            ORDER BY cnt DESC, title ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return {
            "total_jobs": total_jobs,
            "top_skills": [{"name": r[0], "count": int(r[1])} for r in top_skills_rows],
            "top_companies": [{"name": r[0], "count": int(r[1])} for r in top_companies_rows],
            "top_locations": [{"name": r[0], "count": int(r[1])} for r in top_locations_rows],
            "top_titles": [{"name": r[0], "count": int(r[1])} for r in top_titles_rows],
        }
