import json
from pathlib import Path

from src.llm.providers.hf_local import HFLocalExtractor

PROMPT_PATH = Path("src/llm/prompts/jd_extract_v2.txt")

def main():
    jd_paths = list(Path("data/raw/jd_txt").glob("*.txt"))
    json_save_base_path = Path("data/processed/llm/jd_structured")
    if not json_save_base_path.exists():
        json_save_base_path.mkdir(parents=True)

    processed = set(p.stem for p in json_save_base_path.glob("*.json"))

    extractor = HFLocalExtractor(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            device="cuda",
            max_new_tokens=512,
        )

    for jd_path in jd_paths:
        if jd_path.stem in processed:
            print(f"Skipping {jd_path.name}, already processed.")
            continue

        print(f"Processing {jd_path.name}...")
        jd_text = jd_path.read_text()

        prompt = PROMPT_PATH.read_text().replace("{{JOB_DESCRIPTION}}", jd_text)

        out = extractor.extract(prompt)

        if out.get("error"):
            print(f"Error processing {jd_path.name}: {out['error']}")

            # Optionally save the raw output for debugging
            error_save_path = Path("data/processed/llm/errors") / f"{jd_path.stem}_error.txt"
            error_save_path.parent.mkdir(parents=True, exist_ok=True)
            error_save_path.write_text(out.get("raw_output", ""))
            print(f"Saved raw output to {error_save_path}")
            continue

        json_save_path = json_save_base_path / f"{jd_path.stem}.json"
        json_save_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"Saved structured data to {json_save_path}")

if __name__ == "__main__":
    main()
    print(len(list(Path("data/processed/llm/jd_structured").glob("*.json"))), "files processed.")