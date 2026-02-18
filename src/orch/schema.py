from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict


# ----------------------------
# Core schema for extracted job
# ----------------------------

JobStructured = TypedDict(
    "JobStructured",
    {
        "role_title": Optional[str],
        "company": Optional[str],
        "location": Optional[str],
        "employment_type": Optional[str],
        "remote_policy": Optional[str],
        "responsibilities": List[str],
        "requirements": List[str],
        "preferred_qualifications": List[str],
        "skills": List[str],
        "years_experience_min": Optional[int],
        "degree_level": Optional[str],
        "visa_sponsorship": Optional[str],
    },
    total=True,
)


# ----------------------------
# Quality control output
# ----------------------------

class QCResult(TypedDict, total=True):
    status: Literal["pass", "fail"]
    issues: List[str]                 # human-readable reasons
    missing_or_empty: List[str]       # keys that are missing or empty
    coverage: Dict[str, float]        # per-field non-empty indicator (0/1 for single job)
    parse_repaired: bool              # whether repair_json happened
    extractor: Dict[str, Any]         # model meta: local/api model, lora path, etc.


# ----------------------------
# Tool contracts (MCP)
# ----------------------------

# 1) Fetch JD (from DB/files/platform)
class FetchJDInput(TypedDict, total=True):
    job_id: str
    source: Optional[str]  # "handshake", "greenhouse", etc. Optional for now.


class FetchJDOutput(TypedDict, total=True):
    job_id: str
    jd_text: str
    jd_path: Optional[str]     # where saved (if applicable)
    meta: Dict[str, Any]       # e.g., url, timestamp, length, hash


# 2) Local extraction (0.5B baseline or 0.5B+LoRA)
class ExtractLocalInput(TypedDict, total=True):
    job_id: str
    jd_text: str
    prompt_name: str  # e.g., "jd_extract_v2"
    model: str        # e.g., "Qwen/Qwen2.5-0.5B-Instruct"
    lora_path: Optional[str]
    mode: Literal["plain", "chat_lora"]  # plain prompt vs chat-template(+LoRA)


class ExtractLocalOutput(TypedDict, total=True):
    job_id: str
    structured: Optional[JobStructured]   # None if parse fails
    raw_output: str                       # raw completion text
    parse_ok: bool
    parse_repaired: bool
    extractor: Dict[str, Any]             # model meta (mode/model/lora/device)


# 3) API extraction (fallback)
class ExtractAPIInput(TypedDict, total=True):
    job_id: str
    jd_text: str
    prompt_name: str
    provider: Literal["openai_compat"]    # keep abstract
    model: str                            # e.g., "gpt-4o-mini" or "cheap-model"
    temperature: float
    max_tokens: int


class ExtractAPIOutput(TypedDict, total=True):
    job_id: str
    structured: Optional[JobStructured]
    raw_output: str
    parse_ok: bool
    parse_repaired: bool
    usage: Dict[str, Any]                 # tokens/cost if available
    extractor: Dict[str, Any]             # provider/model


# 4) QC validate
class QCValidateInput(TypedDict, total=True):
    job_id: str
    structured: Optional[JobStructured]
    parse_ok: bool
    parse_repaired: bool
    extractor: Dict[str, Any]
    # thresholds (allow per-run override)
    require_keys: List[str]
    require_non_empty_any_of: List[List[str]]  # e.g. [["requirements","responsibilities"]]


class QCValidateOutput(QCResult, total=True):
    job_id: str


# 5) Match analysis (rules now; can add embeddings later)
class MatchInput(TypedDict, total=True):
    job_id: str
    structured: JobStructured
    resume_text: Optional[str]


class MatchOutput(TypedDict, total=True):
    job_id: str
    score: float
    matched_skills: List[str]
    missing_skills: List[str]
    notes: List[str]          # rule-based notes (explainable)


# 6) Report generation (API)
class ReportInput(TypedDict, total=True):
    job_id: str
    structured: JobStructured
    qc: QCResult
    match: Optional[MatchOutput]
    resume_text: Optional[str]
    provider: Literal["openai_compat"]
    model: str
    temperature: float
    max_tokens: int


class ReportOutput(TypedDict, total=True):
    job_id: str
    report_md: str
    usage: Dict[str, Any]
    meta: Dict[str, Any]


# 7) Store artifacts
class StoreInput(TypedDict, total=True):
    job_id: str
    jd_text: Optional[str]
    structured: Optional[JobStructured]
    qc: Optional[QCResult]
    match: Optional[MatchOutput]
    report_md: Optional[str]
    trace: List[Dict[str, Any]]


class StoreOutput(TypedDict, total=True):
    job_id: str
    paths: Dict[str, str]     # jd/structured/qc/report/trace
    meta: Dict[str, Any]


# ----------------------------
# LangGraph State
# ----------------------------

@dataclass
class JobState:
    job_id: str
    source: Optional[str] = None

    jd_text: Optional[str] = None
    jd_path: Optional[str] = None
    jd_meta: Dict[str, Any] = field(default_factory=dict)

    # extraction
    structured: Optional[JobStructured] = None
    raw_output: str = ""
    parse_ok: bool = False
    parse_repaired: bool = False
    extractor_meta: Dict[str, Any] = field(default_factory=dict)

    # qc
    qc: Optional[QCResult] = None

    # optional downstream
    resume_text: Optional[str] = None
    match: Optional[MatchOutput] = None
    report_md: Optional[str] = None

    # trace
    trace: List[Dict[str, Any]] = field(default_factory=list)

    # run config (per job)
    config: Dict[str, Any] = field(default_factory=dict)
