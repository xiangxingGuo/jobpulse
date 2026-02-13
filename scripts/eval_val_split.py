import json
from pathlib import Path
from typing import Dict, Any, List

from src.llm.providers.hf_local import HFLocalExtractor
from src.eval.extraction_metrics import (
    LIST_KEYS, SCALAR_KEYS, non_empty_rate, macro_list_f1
)

VAL_PATH = Path("src/training/datasets/jd_struct_val.jsonl")
OUT_DIR = Path("data/reports")
OUT_REPORT = OUT_DIR / "val_baseline_report.md"
OUT_PRED_DIR = Path("data/processed/llm/val_student_preds")

# IMPORTANT: change only this line later to switch model
STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

    val = load_val()
    prompts = [extract_prompt(ex) for ex in val]
    teacher = [extract_teacher_json(ex) for ex in val]

    extractor = HFLocalExtractor(
        model_name=STUDENT_MODEL,
        device="cuda",
        max_new_tokens=512,
        # lora_path="models/qwen2.5-0.5b-jd-lora"  # <-- change this if you want to test a different model
    )

    student_preds: List[Dict[str, Any]] = []
    failures = 0

    for ex, prompt in zip(val, prompts):
        job_id = ex.get("id", "unknown")
        try:
            pred = extractor.extract(prompt)
            student_preds.append(pred)
            (OUT_PRED_DIR / f"{job_id}.json").write_text(json.dumps(pred, indent=2, ensure_ascii=False))
        except Exception as e:
            failures += 1
            # store empty pred to keep alignment
            student_preds.append({})
            (OUT_PRED_DIR / f"{job_id}.error.txt").write_text(str(e))

    # Coverage / non-empty rates
    teacher_cov = {k: non_empty_rate(teacher, k) for k in (LIST_KEYS + SCALAR_KEYS)}
    student_cov = {k: non_empty_rate(student_preds, k) for k in (LIST_KEYS + SCALAR_KEYS)}

    # Proxy F1 on list keys vs teacher
    list_scores = {k: macro_list_f1(student_preds, teacher, k) for k in LIST_KEYS}

    # Write report
    lines = []
    lines.append("# Val baseline report (student vs teacher)\n")
    lines.append(f"- Val examples: {len(val)}\n")
    lines.append(f"- Student model: `{STUDENT_MODEL}`\n")
    lines.append(f"- Failures: {failures}\n")

    lines.append("\n## Field coverage (non-empty rate)\n")
    lines.append("| Field | Teacher non-empty | Student non-empty |\n")
    lines.append("|---|---:|---:|\n")
    for k in (SCALAR_KEYS + LIST_KEYS):
        lines.append(f"| {k} | {teacher_cov[k]:.2f} | {student_cov[k]:.2f} |\n")

    lines.append("\n## Proxy list F1 (student vs teacher)\n")
    lines.append("> This compares student outputs to teacher outputs (not gold).\n\n")
    lines.append("| Field | Precision | Recall | F1 |\n")
    lines.append("|---|---:|---:|---:|\n")
    for k in LIST_KEYS:
        s = list_scores[k]
        lines.append(f"| {k} | {s.precision:.2f} | {s.recall:.2f} | {s.f1:.2f} |\n")

    OUT_REPORT.write_text("".join(lines), encoding="utf-8")
    print("✅ Wrote report:", OUT_REPORT)
    print("✅ Saved student preds:", OUT_PRED_DIR)


if __name__ == "__main__":
    main()
