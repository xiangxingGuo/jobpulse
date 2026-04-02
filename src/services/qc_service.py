from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QCResult:
    ok: bool
    status: str
    reasons: List[str]
    checks: Dict[str, Any]
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QCService:
    """
    This service validates whether extracted structured data is good enough
    for downstream report generation and fallback routing.
    """

    async def validate(
        self,
        *,
        job_id: str,
        structured: Optional[Dict[str, Any]],
        parse_ok: bool,
        parse_repaired: bool,
        extractor: Optional[Dict[str, Any]] = None,
        require_keys: Optional[List[str]] = None,
        require_non_empty_any_of: Optional[List[List[str]]] = None,
    ) -> QCResult:
        require_keys = require_keys or []
        require_non_empty_any_of = require_non_empty_any_of or []

        reasons: List[str] = []
        checks: Dict[str, Any] = {
            "parse_ok": bool(parse_ok),
            "parse_repaired": bool(parse_repaired),
            "has_structured": structured is not None,
            "missing_required_keys": [],
            "failed_non_empty_groups": [],
            "job_id": job_id,
            "extractor": extractor or {},
        }

        if not parse_ok:
            reasons.append("parse_not_ok")

        if structured is None:
            reasons.append("structured_missing")

        if structured is not None:
            missing_required_keys = [key for key in require_keys if key not in structured]
            checks["missing_required_keys"] = missing_required_keys
            if missing_required_keys:
                reasons.append("missing_required_keys")

            failed_groups: List[List[str]] = []
            for group in require_non_empty_any_of:
                passed = any(self._is_non_empty(structured.get(k)) for k in group)
                if not passed:
                    failed_groups.append(group)

            checks["failed_non_empty_groups"] = failed_groups
            if failed_groups:
                reasons.append("required_non_empty_groups_failed")

        ok = len(reasons) == 0
        status = "pass" if ok else "fail"

        return QCResult(
            ok=ok,
            status=status,
            reasons=reasons,
            checks=checks,
            summary=None if ok else "; ".join(reasons),
        )

    def _is_non_empty(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) > 0
        return True
