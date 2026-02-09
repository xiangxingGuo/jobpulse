from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

DB_PATH = Path("data/db/jobs.db")

SCHEMA = """
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

def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()

def upsert_job(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    cols = [
        "job_id","url","title","company","posted_text","apply_by_text","pay_text",
        "location_text","employment_type","date_range_text","work_auth_text",
        "opt_cpt_text","description"
    ]
    data = {c: row.get(c) for c in cols}
    placeholders = ",".join(["?"] * len(cols))
    assignments = ",".join([f"{c}=excluded.{c}" for c in cols[1:]])  # don't overwrite PK
    sql = f"""
    INSERT INTO jobs ({",".join(cols)}) VALUES ({placeholders})
    ON CONFLICT(job_id) DO UPDATE SET {assignments}
    """
    conn.execute(sql, [data[c] for c in cols])
    conn.commit()

def replace_job_skills(conn: sqlite3.Connection, job_id: str, skills: Iterable[str]) -> None:
    conn.execute("DELETE FROM job_skills WHERE job_id=?", (job_id,))
    conn.executemany(
        "INSERT OR IGNORE INTO job_skills (job_id, skill) VALUES (?, ?)",
        [(job_id, s) for s in sorted(set(skills))]
    )
    conn.commit()

def upsert_job_structured(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    cols = [
        "job_id","model","schema_version","prompt_version",
        "data_json","confidence","error"
    ]
    placeholders = ",".join(["?"] * len(cols))
    assignments = ",".join([f"{c}=excluded.{c}" for c in cols[1:]])
    sql = f"""
    INSERT INTO jobs_structured ({",".join(cols)}) VALUES ({placeholders})
    ON CONFLICT(job_id) DO UPDATE SET {assignments}
    """
    conn.execute(sql, [row.get(c) for c in cols])
    conn.commit()
