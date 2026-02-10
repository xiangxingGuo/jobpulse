import re
from typing import Dict, List, Set

SKILL_MAP: Dict[str, List[str]] = {
    "python": ["python"],
    "sql": ["sql", "postgres", "postgresql", "mysql", "sqlite", "bigquery", "snowflake", "redshift"],
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

def extract_skills_rule_based(text: str) -> Set[str]:
    t = text.lower()
    t = re.sub(r"[\u2013\u2014]", "-", t)
    found: Set[str] = set()

    for skill, kws in SKILL_MAP.items():
        for kw in kws:
            if re.search(rf"(^|[^a-z0-9]){re.escape(kw)}([^a-z0-9]|$)", t):
                found.add(skill)
                break

    return found
