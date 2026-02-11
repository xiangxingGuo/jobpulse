import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROMPT_PATH = Path("src/llm/prompts/jd_extract_v2.txt")

JD_DIR = Path("data/raw/jd_txt")
TEACHER_JSON_DIR = Path("data/processed/llm/Qwen2.5-3B-Instruct/jd_structured")

OUT_DIR = Path("src/training/datasets")
OUT_TRAIN = OUT_DIR / "jd_struct_train.jsonl"
OUT_VAL = OUT_DIR / "jd_struct_val.jsonl"
OUT_GOLD_TEMPLATE = OUT_DIR / "jd_struct_gold_template.jsonl"

SYSTEM_MSG = "You are an information extraction system."

REQUIRED_KEYS = [
    "role_title", "seniority", "location", "employment_type",
    "required_skills", "preferred_skills", "tools",
    "years_experience", "responsibilities", "qualifications"
]

LIST_KEYS = ["required_skills", "preferred_skills", "tools", "responsibilities", "qualifications"]


def load_teacher_json(job_id: str) -> Dict[str, Any]:
    p = TEACHER_JSON_DIR / f"{job_id}.json"
    data = json.loads(p.read_text())
    return data


def validate_schema(d: Dict[str, Any]) -> Tuple[bool, str]:
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        return False, f"missing_keys={missing}"
    for k in LIST_KEYS:
        if not isinstance(d[k], list):
            return False, f"bad_type key={k} type={type(d[k])}"
    return True, ""


def build_prompt(jd_text: str) -> str:
    template = PROMPT_PATH.read_text()
    return template.replace("{{JOB_DESCRIPTION}}", jd_text)


def make_example(job_id: str, jd_text: str, teacher: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_prompt(jd_text)
    assistant = json.dumps(teacher, ensure_ascii=False)  # keep strict JSON string
    return {
        "id": job_id,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ]
    }


def main(seed: int = 42, val_size: int = 12) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jd_paths = sorted(JD_DIR.glob("*.txt"))
    if not jd_paths:
        raise FileNotFoundError(f"No JD files found in {JD_DIR}")

    # only keep ids that have teacher structured json
    pairs: List[Tuple[str, Path]] = []
    skipped = []
    for p in jd_paths:
        job_id = p.stem
        teacher_path = TEACHER_JSON_DIR / f"{job_id}.json"
        if not teacher_path.exists():
            skipped.append((job_id, "no_teacher_json"))
            continue
        pairs.append((job_id, p))

    if not pairs:
        raise RuntimeError("No (JD, teacher_json) pairs found. Check your directories.")

    random.seed(seed)
    random.shuffle(pairs)

    val_size = min(val_size, max(1, len(pairs) // 5))  # cap val to ~20% if dataset small
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    train_lines = []
    val_lines = []
    gold_lines = []

    bad = []

    for job_id, jd_path in train_pairs:
        jd_text = jd_path.read_text()
        teacher = load_teacher_json(job_id)
        ok, msg = validate_schema(teacher)
        if not ok:
            bad.append((job_id, msg))
            continue
        train_lines.append(make_example(job_id, jd_text, teacher))

    for job_id, jd_path in val_pairs:
        jd_text = jd_path.read_text()
        teacher = load_teacher_json(job_id)
        ok, msg = validate_schema(teacher)
        if not ok:
            bad.append((job_id, msg))
            continue
        val_lines.append(make_example(job_id, jd_text, teacher))

    # gold template: same as val candidates, but assistant left empty for manual labeling
    for job_id, jd_path in val_pairs:
        jd_text = jd_path.read_text()
        gold_lines.append({
            "id": job_id,
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": build_prompt(jd_text)},
                {"role": "assistant", "content": ""}  # you fill this manually later
            ],
            "notes": "Fill assistant with STRICT JSON matching the schema."
        })

    # write jsonl
    OUT_TRAIN.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in train_lines) + "\n")
    OUT_VAL.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in val_lines) + "\n")
    OUT_GOLD_TEMPLATE.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in gold_lines) + "\n")

    print("✅ Dataset built")
    print(f"Train: {len(train_lines)} -> {OUT_TRAIN}")
    print(f"Val:   {len(val_lines)} -> {OUT_VAL}")
    print(f"Gold template (from val split): {len(gold_lines)} -> {OUT_GOLD_TEMPLATE}")

    if skipped:
        print(f"⚠️ Skipped (no teacher json): {len(skipped)}")

    if bad:
        print(f"⚠️ Bad schema outputs: {len(bad)}")
        bad_path = OUT_DIR / "bad_teacher_outputs.txt"
        bad_path.write_text("\n".join(f"{jid}\t{reason}" for jid, reason in bad) + "\n")
        print("Details:", bad_path)


if __name__ == "__main__":
    main()
