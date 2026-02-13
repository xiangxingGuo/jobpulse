# src/llm/providers/base.py
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


# ---------- Data structures ----------

@dataclass
class ExtractionResult:
    """
    Unified extraction result.
    - data: parsed JSON dict if successful, else None
    - raw_output: model raw text completion (useful for debugging)
    - error: error message if failed, else None
    """
    data: Optional[Dict[str, Any]]
    raw_output: str = ""
    error: Optional[str] = None


# ---------- Base interface + shared utilities ----------

class BaseExtractor(ABC):
    """
    Base class for all extractors. Subclasses implement _generate(prompt)->str
    which returns the model completion text (ideally JSON only).
    """

    def __init__(
        self,
        required_keys: Optional[Sequence[str]] = None,
        list_keys: Optional[Sequence[str]] = None,
    ) -> None:
        self.required_keys = list(required_keys) if required_keys is not None else []
        self.list_keys = list(list_keys) if list_keys is not None else []

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        """
        Generate model completion text for a given prompt.
        Should return the *completion only* (not including the prompt),
        but BaseExtractor's parsing can tolerate some extra text.
        """
        raise NotImplementedError

    def extract(self, prompt: str) -> Dict[str, Any]:
        """
        Convenience API: returns only the parsed dict or raises.
        """
        result = self.extract_with_result(prompt)

        return result

    def extract_with_result(self, prompt: str) -> ExtractionResult:
        """
        Robust extraction: generate -> parse JSON -> validate schema.
        Never throws unless subclass _generate throws unexpectedly.
        """
        raw = ""
        try:
            raw = self._generate(prompt).strip()
            data = self.parse_json(raw)
            self.validate_schema(data)
            return ExtractionResult(data=data, raw_output=raw, error=None)
        except Exception as e:
            return ExtractionResult(data=None, raw_output=raw, error=str(e))

    # ---------- JSON parsing ----------

    def parse_json(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON dict from model output.
        Fast path: json.loads(text)
        Fallback: extract last valid JSON object from text.
        """
        # Fast path
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"Parsed JSON is not an object/dict: {type(obj)}")
            return obj
        except json.JSONDecodeError:
            # Fallback
            obj = self._extract_last_json_object(text)
            if not isinstance(obj, dict):
                raise ValueError(f"Parsed JSON is not an object/dict: {type(obj)}")
            return obj
    

    def _extract_last_json_object(self, text: str) -> Any:
        """
        Slow but robust fallback: find all top-level {...} blocks by brace matching,
        then parse from last to first until one succeeds.
        """
        candidates: List[str] = []
        depth = 0
        start: Optional[int] = None

        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(text[start : i + 1])
                        start = None

        if not candidates:
            raise ValueError("No JSON object found in model output")

        last_err: Optional[Exception] = None
        for s in reversed(candidates):
            try:
                return json.loads(s)
            except Exception as e:
                last_err = e
                continue

        raise ValueError(f"Found JSON-like blocks but none parsed: {last_err}")

    # ---------- Schema validation ----------

    def validate_schema(self, data: Dict[str, Any]) -> None:
        """
        Validate presence of required keys and types of list fields.
        If no required_keys/list_keys configured, does nothing.
        """
        if self.required_keys:
            missing = [k for k in self.required_keys if k not in data]
            if missing:
                raise ValueError(f"Missing required keys: {missing}")

        if self.list_keys:
            bad = []
            for k in self.list_keys:
                if k in data and not isinstance(data[k], list):
                    bad.append((k, type(data[k]).__name__))
            if bad:
                raise ValueError(f"List-typed keys have wrong types: {bad}")
