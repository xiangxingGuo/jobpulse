import json
from pathlib import Path
import tqdm

from src.llm.providers.hf_plain import HFPlainExtractor

from src.eval.extraction_metrics import (
    REQUIRED_KEYS, LIST_KEYS
)


PROMPT_PATH = Path("src/llm/prompts/jd_extract_v2.txt")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

def main():
    jd_paths = list(Path("data/raw/jd_txt").glob("*.txt"))
    tag = f"{MODEL_NAME.split('/')[-1]}"
    json_save_base_path = Path(f"data/processed/llm/{tag}/jd_structured")

    if not json_save_base_path.exists():
        json_save_base_path.mkdir(parents=True)
    
    processed = set(p.stem for p in json_save_base_path.glob("*.json"))

    extractor = HFPlainExtractor(
            model_name=MODEL_NAME,
            device="cuda",
            max_new_tokens=512,
             required_keys=REQUIRED_KEYS,
             list_keys=LIST_KEYS
        )
    
    jd_paths = [p for p in jd_paths if p.stem not in processed]

    for jd_path in tqdm.tqdm(jd_paths, desc="Processing job descriptions"):
        tqdm.tqdm.write(f"Processing {jd_path.name}...")

        jd_text = jd_path.read_text()

        prompt = PROMPT_PATH.read_text().replace("{{JOB_DESCRIPTION}}", jd_text)

        # safe check for placeholder
        assert "{{JOB_DESCRIPTION}}" in PROMPT_PATH.read_text(), "Prompt template must contain {{JOB_DESCRIPTION}} placeholder"

        result = extractor.extract(prompt)

        if result.error:
            tqdm.tqdm.write(f"Error processing {jd_path.name}: {result.error}")

            # Optionally save the raw output for debugging
            error_save_path = Path(f"data/processed/llm/{tag}/errors") / f"{jd_path.stem}_error.txt"
            error_save_path.parent.mkdir(parents=True, exist_ok=True)
            error_save_path.write_text(result.raw_output or "")
            tqdm.tqdm.write(f"Saved raw output to {error_save_path}")
            continue

        json_save_path = json_save_base_path / f"{jd_path.stem}.json"
        json_save_path.write_text(json.dumps(result.data, indent=2, ensure_ascii=False))
        tqdm.tqdm.write(f"Saved structured data to {json_save_path}")
    
    print(len(list(json_save_base_path.glob("*.json"))), "files processed.")


if __name__ == "__main__":
    main()