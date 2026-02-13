import json
from pathlib import Path

from src.llm.providers.hf_local import HFLocalExtractor

PROMPT_PATH = Path("src/llm/prompts/jd_extract_v1.txt")

def main():
    # only one for smoke test
    jd_path = Path("data/raw/jd_txt/10704289.txt")
    jd_text = jd_path.read_text()

    prompt = PROMPT_PATH.read_text().replace("{{JOB_DESCRIPTION}}", jd_text)

    extractor = HFLocalExtractor(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="cuda",
        max_new_tokens=512,
    )

    out = extractor.extract(prompt)

    # 最关键：结构校验（字段必须齐全）
    required_keys = [
        "role_title","seniority","location","employment_type",
        "required_skills","preferred_skills","tools",
        "years_experience","responsibilities","qualifications"
    ]
    missing = [k for k in required_keys if k not in out]
    if missing:
        raise AssertionError(f"Missing keys: {missing}")

    # list 字段必须是 list
    for k in ["required_skills","preferred_skills","tools","responsibilities","qualifications"]:
        if not isinstance(out[k], list):
            raise AssertionError(f"{k} must be list, got {type(out[k])}")

    debug_path = Path("data/raw/llm_debug") / "smoke_baseline.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("✅ Smoke test passed. Output saved to:", debug_path)

if __name__ == "__main__":
    main()
