from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.orch.schema import JobStructured, QCValidateOutput


def _is_non_empty(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, list):
        return len(v) > 0
    return True


def qc_validate(
    job_id: str,
    structured: Optional[JobStructured],
    parse_ok: bool,
    parse_repaired: bool,
    extractor: Dict[str, Any],
    require_keys: List[str],
    require_non_empty_any_of: List[List[str]],
) -> QCValidateOutput:
    """
    Validate schema + minimal coverage gates.
    """
    issues: List[str] = []
    missing_or_empty: List[str] = []
    coverage: Dict[str, float] = {}

    if not parse_ok or structured is None:
        return {
            "job_id": job_id,
            "status": "fail",
            "issues": ["parse_failed"],
            "missing_or_empty": require_keys,
            "coverage": {},
            "parse_repaired": parse_repaired,
            "extractor": extractor,
        }

    # required keys present + non-empty
    for k in require_keys:
        v = structured.get(k)  # type: ignore
        ok = _is_non_empty(v)
        coverage[k] = 1.0 if ok else 0.0
        if not ok:
            missing_or_empty.append(k)

    # at least one group satisfies "any_of"
    for group in require_non_empty_any_of:
        if not any(_is_non_empty(structured.get(k)) for k in group):  # type: ignore
            issues.append(f"low_coverage_any_of:{group}")

    if missing_or_empty:
        issues.append("missing_or_empty_required")

    status = "pass" if not issues else "fail"
    return {
        "job_id": job_id,
        "status": status,
        "issues": issues,
        "missing_or_empty": missing_or_empty,
        "coverage": coverage,
        "parse_repaired": parse_repaired,
        "extractor": extractor,
    }
