from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence
from tqdm import tqdm
import os

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.eval.extraction_metrics import (
    LIST_KEYS, SCALAR_KEYS, non_empty_rate, macro_list_f1, REQUIRED_KEYS
)
from src.llm.providers.base import BaseExtractor

VAL_PATH = Path("src/training/datasets/jd_struct_val.jsonl")

def load_val() -> List[Dict[str, Any]]:
    lines = [json.loads(x) for x in VAL_PATH.read_text().splitlines() if x.strip()]
    return lines


def extract_teacher_json(example: Dict[str, Any]) -> Dict[str, Any]:
    # messages[-1] is assistant with teacher JSON string
    teacher_str = example["messages"][-1]["content"]
    return json.loads(teacher_str)


def extract_prompt(example: Dict[str, Any]) -> str:
    # messages[1] is user with prompt (already includes JD)
    return example["messages"][1]["content"]


class Eval_Base(ABC):
    def __init__(
            self,
            extractor: BaseExtractor,
            OUT_DIR: Path,
            OUT_PRED_DIR: Path,
            val: List[Dict[str, Any]],
            prompts: List[str],
    ) -> None:
        self.extractor = extractor
        self.OUT_DIR = OUT_DIR
        self.OUT_PRED_DIR = OUT_PRED_DIR
        self.val = val
        self.prompts = prompts
    
    # obtain predictions
    def obtain_predictions(self):
        preds: List[Dict[str, Any]] = []
        failures = 0

        os.makedirs(self.OUT_PRED_DIR / 'errors', exist_ok=True)
        os.makedirs(self.OUT_PRED_DIR / 'jd_structured', exist_ok=True)

        # for ex, prompt in zip(self.val, self.prompts):
        for ex, prompt in tqdm(zip(self.val, self.prompts), total=len(self.prompts)):
            job_id = ex.get("id", "unknown")

            pred = self.get_prediction(prompt)

            if pred.error:
                failures += 1
                preds.append({})
                (self.OUT_PRED_DIR / 'errors' / f"{job_id}.error.txt").write_text(pred.error)
                (self.OUT_PRED_DIR / 'errors' / f"{job_id}.raw.txt").write_text(pred.raw_output)
            
            else:
                preds.append(pred.data)
                (self.OUT_PRED_DIR /'jd_structured' /f"{job_id}.json").write_text(
                    json.dumps(pred.data, indent=2, ensure_ascii=False),
                )
        
        return preds, failures
            

    # get one prediction for one prompt
    def get_prediction(self, prompt: str) -> str:
        
        pred = self.extractor.extract_with_result(prompt)

        return pred

    # Coverage / non-empty rates
    def compute_coverage(self, preds: List[Dict[str, Any]]):
        pred_cov = {k: non_empty_rate(preds, k) for k in (LIST_KEYS + SCALAR_KEYS)}
        return pred_cov
    