from eval_base import Eval_Base, load_val, extract_prompt
from src.llm.providers.hf_chat_lora import HFChatLoRAExtractor
import json
from pathlib import Path
from typing import Dict, Any, List
from src.eval.extraction_metrics import REQUIRED_KEYS, LIST_KEYS
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/reports")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA weights directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    tag = args.model_name.split("/")[-1]

    OUT_DIR = Path("data/reports")
    OUT_PRED_DIR = Path(f"data/processed/llm/{tag}/val_student_preds")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

    extractor = HFChatLoRAExtractor(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        required_keys=REQUIRED_KEYS,
        list_keys=LIST_KEYS,
        lora_path=args.lora_path,
    )

    val = load_val()
    prompts = [extract_prompt(ex) for ex in val]
    eval = Eval_Base(
        extractor=extractor,
        OUT_DIR=OUT_DIR,
        OUT_PRED_DIR=OUT_PRED_DIR,
        val=val,
        prompts=prompts,
    )

    preds, failures = eval.obtain_predictions()
    print(f"Obtained predictions for {len(preds)} examples with {failures} failures.")

    conv = eval.compute_coverage(preds)
    print("Prediction coverage:", conv)

if __name__ == "__main__":
    main()

