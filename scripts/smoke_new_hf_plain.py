import json
from pathlib import Path
import tqdm

from src.llm.providers.hf_plain import HFPlainExtractor

REQUIRED_KEYS = [
    "role_title", "company", "location", "employment_type",
    "remote_policy", "responsibilities", "requirements",
    "preferred_qualifications", "skills", "years_experience_min",
    "degree_level", "visa_sponsorship"
]

LIST_KEYS = ["responsibilities", "requirements", 
             "preferred_qualifications", "skills"]

PROMPT_PATH = Path("src/llm/prompts/jd_extract_v2.txt")

def main():
    # only one for smoke test
    # data/raw/jd_txt/1035622.txt
    jd_path = Path("data/raw/jd_txt/1035622.txt")
    jd_text = jd_path.read_text()

    prompt = PROMPT_PATH.read_text().replace("{{JOB_DESCRIPTION}}", jd_text)

    extractor = HFPlainExtractor(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            device="cuda",
            max_new_tokens=512,
             required_keys=REQUIRED_KEYS,
             list_keys=LIST_KEYS
        )
    
    out = extractor.extract(prompt)

    print(json.dumps(out, indent=2, ensure_ascii=False))

    pass

if __name__ == "__main__":
    main()

