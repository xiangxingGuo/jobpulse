import re
from typing import Iterable


DEFAULT_DROP_PATTERNS: list[str] = [
    r"^\s*Apply\s*$",
    r"^\s*Apply by\s*.*$",
    r"^\s*Posted\s+.*$",
    r"^\s*At a glance\s*$",
    r"^\s*Show more\s*$",
    r"^\s*See more\s*$",
    r"^\s*More\s*$",
    r"^\s*Handshake\s*$",
]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_noisy_lines(text: str, drop_patterns: Iterable[str] = DEFAULT_DROP_PATTERNS) -> str:
    pats = [re.compile(p, re.IGNORECASE) for p in drop_patterns]
    kept: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if any(p.search(s) for p in pats):
            continue
        kept.append(line)
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def clean_jd(raw_text: str) -> str:
    text = normalize_whitespace(raw_text)
    text = drop_noisy_lines(text)
    return text
