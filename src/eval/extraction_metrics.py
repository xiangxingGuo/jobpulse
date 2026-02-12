from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

REQUIRED_KEYS = [
    "role_title", "company", "location", "employment_type",
    "remote_policy", "responsibilities", "requirements",
    "preferred_qualifications", "skills", "years_experience_min",
    "degree_level", "visa_sponsorship"
]

LIST_KEYS = ["responsibilities", "requirements", 
             "preferred_qualifications", "skills"]

SCALAR_KEYS = [k for k in REQUIRED_KEYS if k not in LIST_KEYS]


def _norm_item(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    return " ".join(s.split())


def _to_set(lst: Any) -> Set[str]:
    if not isinstance(lst, list):
        return set()
    return { _norm_item(x) for x in lst if _norm_item(x) }


def set_f1(pred: Any, gold: Any) -> Tuple[float, float, float]:
    p = _to_set(pred)
    g = _to_set(gold)
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p and g:
        return 0.0, 0.0, 0.0
    if p and not g:
        return 0.0, 0.0, 0.0

    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def non_empty_rate(examples: List[Dict[str, Any]], key: str) -> float:
    if not examples:
        return 0.0
    non_empty = 0
    for ex in examples:
        v = ex.get(key)
        if isinstance(v, list):
            if len(v) > 0:
                non_empty += 1
        else:
            if str(v or "").strip():
                non_empty += 1
    return non_empty / len(examples)


@dataclass
class ListKeyScores:
    precision: float
    recall: float
    f1: float


def macro_list_f1(preds: List[Dict[str, Any]], golds: List[Dict[str, Any]], key: str) -> ListKeyScores:
    assert len(preds) == len(golds)
    ps, rs, fs = 0.0, 0.0, 0.0
    n = len(preds)
    for p, g in zip(preds, golds):
        prec, rec, f1 = set_f1(p.get(key), g.get(key))
        ps += prec
        rs += rec
        fs += f1
    if n == 0:
        return ListKeyScores(0.0, 0.0, 0.0)
    return ListKeyScores(ps / n, rs / n, fs / n)
