from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def extract_jsonish_tail(text: str) -> str:
    t = text or ""
    i_obj = t.find("{")
    i_arr = t.find("[")
    if i_obj == -1 and i_arr == -1:
        return t.strip()
    if i_obj == -1:
        start = i_arr
    elif i_arr == -1:
        start = i_obj
    else:
        start = min(i_obj, i_arr)
    return t[start:].strip()


def truncate_to_last_balanced(text: str) -> Optional[str]:
    """
    If output contains JSON followed by extra text, truncate at the last point
    where brackets are balanced (outside strings).
    """
    t = text or ""
    stack: List[str] = []
    in_str = False
    esc = False
    started = False
    last_balanced_end: Optional[int] = None

    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
            started = True
        elif ch in "}]":
            if stack:
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()

        if started and not in_str and not stack:
            last_balanced_end = i + 1

    if last_balanced_end is not None:
        return t[:last_balanced_end].strip()
    return None


def repair_brackets(text: str, max_append: int = 256) -> str:
    """
    Best-effort:
    1) strip fences
    2) keep json-ish tail
    3) truncate to last balanced (fixes "Extra data")
    4) otherwise append missing closers (fixes missing } / ])
    """
    t = strip_code_fences(text)
    t = extract_jsonish_tail(t)

    truncated = truncate_to_last_balanced(t)
    if truncated:
        return truncated

    stack: List[str] = []
    in_str = False
    esc = False

    for ch in t:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()

    closes = []
    for opener in reversed(stack):
        closes.append("}" if opener == "{" else "]")
    if closes:
        t = t + "".join(closes)[:max_append]
    return t.strip()


def parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """
    Returns: (obj_or_none, repaired_flag, used_text)
    - obj_or_none: dict if parse succeeded and is a JSON object; else None
    - repaired_flag: True if we had to modify input (repair/truncate/etc.)
    - used_text: the exact string we attempted to json.loads on the final attempt
    """
    raw = (text or "").strip()

    # First try as-is
    try:
        obj = json.loads(raw)
        return (obj if isinstance(obj, dict) else None, False, raw)
    except Exception:
        pass

    # Repair/truncate
    repaired = repair_brackets(raw)
    try:
        obj = json.loads(repaired)
        return (obj if isinstance(obj, dict) else None, True, repaired)
    except Exception:
        return (None, True, repaired)
