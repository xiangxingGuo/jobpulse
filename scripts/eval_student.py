import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.llm.providers.hf_local import HFLocalExtractor
from src.eval.extraction_metrics import (
    LIST_KEYS, SCALAR_KEYS, non_empty_rate, macro_list_f1
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def extract_teacher_json(example: Dict[str, Any]) -> Dict[str, Any]:
    teacher_str = example["messages"][-1]["content"]
    return json.loads(teacher_str)


def extract_prompt(example: Dict[str, Any]) -> str:
    return example["messages"][1]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="src/training/datasets/jd_struct_val.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lora", default="", help="LoRA adapter path (optional)")
    ap.add_argument("--tag", required=True, help="Run tag used for report/preds filenames")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    val_path = Path(args.val)
    tag = args.tag

    report_dir = Path("data/reports/eval")
    pred_dir = Path("data/processed/llm/eval_preds") / tag
    report_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    out_report = report_dir / f"{tag}.md"

    val = load_jsonl(val_path)
    prompts = [extract_prompt(ex) for ex in val]
    teacher = [extract_teacher_json(ex) for ex in val]

    lora_path: Optional[str] = args.lora.strip() or None
    extractor = HFLocalExtractor(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        lora_path=lora_path,
    )

    student_preds: List[Dict[str, Any]] = []
    failures = 0

    for ex, prompt in zip(val, prompts):
        job_id = ex.get("id", "unknown")
        try:
            pred = extractor.extract(prompt)
            student_preds.append(pred)
            (pred_dir / f"{job_id}.json").write_text(
                json.dumps(pred, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            failures += 1
            student_preds.append({})
            (pred_dir / f"{job_id}.error.txt").write_text(str(e), encoding="utf-8")

    teacher_cov = {k: non_empty_rate(teacher, k) for k in (LIST_KEYS + SCALAR_KEYS)}
    student_cov = {k: non_empty_rate(student_preds, k) for k in (LIST_KEYS + SCALAR_KEYS)}
    list_scores = {k: macro_list_f1(student_preds, teacher, k) for k in LIST_KEYS}

    lines = []
    lines.append("# Val report (student vs teacher)\n")
    lines.append(f"- Tag: `{tag}`\n")
    lines.append(f"- Val examples: {len(val)}\n")
    lines.append(f"- Student model: `{args.model}`\n")
    lines.append(f"- LoRA: `{lora_path or 'NONE'}`\n")
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

    out_report.write_text("".join(lines), encoding="utf-8")
    print("✅ Wrote report:", out_report)
    print("✅ Saved student preds:", pred_dir)


if __name__ == "__main__":
    main()
