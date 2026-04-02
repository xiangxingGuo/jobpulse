from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set

from src.llm.providers.hf_local import HFLocalExtractor

PROMPT_PATH = Path("src/llm/prompts/jd_extract_v1.txt")

SKILL_MAP: Dict[str, List[str]] = {
    "python": ["python"],
    "sql": [
        "sql",
        "postgres",
        "postgresql",
        "mysql",
        "sqlite",
        "bigquery",
        "snowflake",
        "redshift",
    ],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow", "tf"],
    "sklearn": ["scikit-learn", "sklearn"],
    "xgboost": ["xgboost"],
    "spark": ["spark", "pyspark"],
    "airflow": ["airflow"],
    "dbt": ["dbt"],
    "kafka": ["kafka"],
    "aws": ["aws", "s3", "ec2", "lambda", "sagemaker"],
    "gcp": ["gcp", "vertex", "bigquery"],
    "azure": ["azure"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "fastapi": ["fastapi"],
    "mlops": ["mlops", "ml flow", "mlflow", "kubeflow"],
    "llm": ["llm", "large language model", "gpt", "transformer"],
    "nlp": ["nlp", "natural language processing"],
}


def extract_skills(text: str) -> Set[str]:
    t = text.lower()
    t = re.sub(r"[\u2013\u2014]", "-", t)  # en/em dash
    found: Set[str] = set()
    for skill, kws in SKILL_MAP.items():
        for kw in kws:
            # word-boundary-ish matching
            if re.search(rf"(^|[^a-z0-9]){re.escape(kw)}([^a-z0-9]|$)", t):
                found.add(skill)
                break
    return found


def extract_job_baseline(job_id: str, jd_text: str) -> dict:
    prompt_template = PROMPT_PATH.read_text()
    prompt = prompt_template.replace("{{JOB_DESCRIPTION}}", jd_text)

    extractor = HFLocalExtractor(model_name="Qwen/Qwen2.5-3B-Instruct", device="cuda")

    result = extractor.extract(prompt)

    # debug dump
    debug_path = Path("data/raw/llm_debug") / f"{job_id}_baseline.json"
    debug_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    return result
