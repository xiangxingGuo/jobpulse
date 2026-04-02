"""
This module currently serves MCP/tool-side schemas and legacy typed models.
LangGraph runtime state is currently defined and enforced in graph.py.
"""

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
    issues: List[str]  # human-readable reasons
    missing_or_empty: List[str]  # keys that are missing or empty
    coverage: Dict[str, float]  # per-field non-empty indicator (0/1 for single job)
    parse_repaired: bool  # whether repair_json happened
    extractor: Dict[str, Any]  # model meta: local/api model, lora path, etc.


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
    jd_path: Optional[str]  # where saved (if applicable)
    meta: Dict[str, Any]  # e.g., url, timestamp, length, hash


# 2) Local extraction (0.5B baseline or 0.5B+LoRA)
class ExtractLocalInput(TypedDict, total=True):
    job_id: str
    jd_text: str
    prompt_name: str  # e.g., "jd_extract_v2"
    model: str  # e.g., "Qwen/Qwen2.5-0.5B-Instruct"
    lora_path: Optional[str]
    mode: Literal["plain", "chat_lora"]  # plain prompt vs chat-template(+LoRA)
    device: str  # "cuda" or "cpu" (allow override for testing on CPU)
    max_new_tokens: int


class ExtractLocalOutput(TypedDict, total=True):
    job_id: str
    structured: Optional[JobStructured]  # None if parse fails
    raw_output: str  # raw completion text
    parse_ok: bool
    parse_repaired: bool
    extractor: Dict[str, Any]  # model meta (mode/model/lora/device)


# 3) API extraction (fallback)
class ExtractAPIInput(TypedDict, total=True):
    job_id: str
    jd_text: str
    prompt_name: str
    provider: Literal["openai", "nvidia"]  # OpenAI-compatible providers
    model: str  # e.g., "gpt-4o-mini" or "cheap-model"
    temperature: float
    max_tokens: int
    thinking: Literal["auto", "disabled", "enabled"]


class ExtractAPIOutput(TypedDict, total=True):
    job_id: str
    structured: Optional[JobStructured]
    raw_output: str
    parse_ok: bool
    parse_repaired: bool
    usage: Dict[str, Any]  # tokens/cost if available
    extractor: Dict[str, Any]  # provider/model


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
    notes: List[str]  # rule-based notes (explainable)


# 6) Report generation (API)
class ReportInput(TypedDict, total=True):
    job_id: str
    structured: JobStructured
    qc: QCResult
    match: Optional[MatchOutput]
    resume_text: Optional[str]
    provider: Literal["openai", "nvidia"]  # OpenAI-compatible providers
    model: str
    temperature: float
    max_tokens: int
    thinking: Literal["auto", "disabled", "enabled"]


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
    paths: Dict[str, str]  # jd/structured/qc/report/trace
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


# ----------------------------
# LangGraph v2 State
# ----------------------------


@dataclass
class RunMeta:
    run_id: str
    workflow: str
    status: str
    started_at: str
    ended_at: Optional[str] = None
    route: Optional[str] = None
    entrypoint: str = "langgraph"


@dataclass
class InputState:
    job_id: str
    source: str = "handshake"
    resume_text: Optional[str] = None


@dataclass
class RoutingConfig:
    primary_mode: Literal["local", "api"] = "api"
    fallback_mode: Literal["api", "local"] = "api"
    fallback_enabled: bool = True
    local_enabled: bool = False


@dataclass
class ModelConfig:
    prompt_name: str = "jd_extract_v2"

    extract_provider: Literal["openai", "nvidia"] = "openai"
    extract_model: Optional[str] = None

    report_provider: Literal["openai", "nvidia"] = "openai"
    report_model: Optional[str] = None

    local_mode: str = "chat_lora"
    local_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    local_lora_path: Optional[str] = None


@dataclass
class QCPolicy:
    require_keys: List[str] = field(
        default_factory=lambda: ["role_title", "company", "requirements", "responsibilities"]
    )
    require_non_empty_any_of: List[List[str]] = field(
        default_factory=lambda: [["requirements", "responsibilities"]]
    )


@dataclass
class FeatureFlags:
    enable_skill_gap: bool = False


@dataclass
class JobData:
    jd_text: Optional[str] = None
    jd_path: Optional[str] = None
    jd_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionAttempt:
    stage: str  # primary / fallback
    mode: Literal["local", "api"]
    structured: Optional[JobStructured]
    raw_output: str
    parse_ok: bool
    parse_repaired: bool
    extractor: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ExtractionState:
    attempts: List[ExtractionAttempt] = field(default_factory=list)
    selected_attempt: Optional[int] = None


@dataclass
class QCAttempt:
    stage: str  # primary / fallback
    ok: bool
    status: Literal["pass", "fail"]
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None


@dataclass
class QCState:
    attempts: List[QCAttempt] = field(default_factory=list)
    selected_attempt: Optional[int] = None


@dataclass
class ReportState:
    report_md: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactState:
    base_dir: Optional[str] = None
    paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class JobRunState:
    run: RunMeta
    input: InputState
    config_routing: RoutingConfig = field(default_factory=RoutingConfig)
    config_models: ModelConfig = field(default_factory=ModelConfig)
    qc_policy: QCPolicy = field(default_factory=QCPolicy)
    features: FeatureFlags = field(default_factory=FeatureFlags)

    job: JobData = field(default_factory=JobData)
    extraction: ExtractionState = field(default_factory=ExtractionState)
    qc_state: QCState = field(default_factory=QCState)
    report_state: ReportState = field(default_factory=ReportState)
    artifacts: ArtifactState = field(default_factory=ArtifactState)

    trace: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=lambda: {"node_ms": {}})
    errors: List[Dict[str, Any]] = field(default_factory=list)
