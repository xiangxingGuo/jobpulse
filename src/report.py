from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import List, Tuple

def build_report(conn: sqlite3.Connection, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    recent = conn.execute("SELECT COUNT(*) FROM jobs WHERE scraped_at_utc >= datetime('now','-7 day')").fetchone()[0]

    top_skills: List[Tuple[str,int]] = conn.execute("""
        SELECT skill, COUNT(*) as c
        FROM job_skills
        GROUP BY skill
        ORDER BY c DESC
        LIMIT 20
    """).fetchall()

    top_locations = conn.execute("""
        SELECT COALESCE(location_text,'(unknown)') as loc, COUNT(*) as c
        FROM jobs
        GROUP BY loc
        ORDER BY c DESC
        LIMIT 10
    """).fetchall()

    opt = conn.execute("""
        SELECT COUNT(*) FROM jobs
        WHERE LOWER(COALESCE(opt_cpt_text,'')) LIKE '%opt%'
           OR LOWER(COALESCE(description,'')) LIKE '%opt/cpt%'
           OR LOWER(COALESCE(description,'')) LIKE '%cpt%'
    """).fetchone()[0]

    remote = conn.execute("""
        SELECT COUNT(*) FROM jobs
        WHERE LOWER(COALESCE(location_text,'')) LIKE '%remote%'
           OR LOWER(COALESCE(description,'')) LIKE '%work from home%'
    """).fetchone()[0]

    lines = []
    lines.append("# JobPulse Weekly Snapshot\n")
    lines.append(f"- Total jobs in DB: **{total}**")
    lines.append(f"- Jobs scraped in last 7 days: **{recent}**")
    lines.append(f"- Jobs mentioning OPT/CPT: **{opt}**")
    lines.append(f"- Jobs mentioning Remote/WFH: **{remote}**\n")

    lines.append("## Top Skills (from JD text)\n")
    for s, c in top_skills:
        lines.append(f"- {s}: {c}")
    lines.append("")

    lines.append("## Top Locations (raw)\n")
    for loc, c in top_locations:
        lines.append(f"- {loc}: {c}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
