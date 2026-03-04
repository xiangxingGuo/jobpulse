from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any, Optional, List, Set
from datetime import datetime, timezone

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

def end_run(run_id: str, summary: Dict[str, Any], elapsed_sec: float, slo_met: Optional[bool]) -> None:
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

def upsert_job(row: Dict[str, Any], scrape_status: str = "success", scrape_error: Optional[str] = None) -> None:
    cols = [
        "job_id","url","title","company","posted_text","apply_by_text","pay_text",
        "location_text","employment_type","date_range_text","work_auth_text",
        "opt_cpt_text","description",
        "scrape_status","scrape_error","content_hash","last_seen_at_utc",
    ]
    data = {c: row.get(c) for c in cols}

    # Set operational fields
    data["scrape_status"] = scrape_status
    data["scrape_error"] = scrape_error
    data["last_seen_at_utc"] = data.get("last_seen_at_utc") or _utc_now_iso()

    # content_hash for idempotency
    desc = (row.get("description") or "")
    data["content_hash"] = row.get("content_hash") or (None if not desc else __import__("hashlib").sha1(desc.encode("utf-8", errors="ignore")).hexdigest())

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
        "job_id","model","schema_version","prompt_version",
        "data_json","confidence","error"
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