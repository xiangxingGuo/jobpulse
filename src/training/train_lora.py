import os
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model


# ====== Config (change only these) ======
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_PATH = "src/training/datasets/jd_struct_train.jsonl"
VAL_PATH = "src/training/datasets/jd_struct_val.jsonl"

OUT_DIR = "models/qwen2.5-0.5b-jd-lora"
MAX_LENGTH = 512  # keep conservative for 8GB
EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8
WARMUP_RATIO = 0.03
SEED = 42
# =======================================


def messages_to_text(example, tokenizer):
    """
    Convert messages format to a single chat string using the tokenizer's chat template (if available).
    """
    msgs = example["messages"]
    # Ensure assistant content exists (train/val should have it)
    if not msgs or msgs[-1]["role"] != "assistant":
        raise ValueError("Invalid example: messages must end with assistant")

    # Use chat template if model/tokenizer supports it
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    else:
        # Fallback: simple concatenation
        parts = []
        for m in msgs:
            parts.append(f"{m['role'].upper()}:\n{m['content']}\n")
        text = "\n".join(parts)
    return {"text": text}


def tokenize_fn(example, tokenizer):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None,
    )
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config: keep small and stable
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # common for Qwen
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})

    # Convert messages -> text
    ds = ds.map(lambda x: messages_to_text(x, tokenizer), remove_columns=ds["train"].column_names)

    # Tokenize
    ds = ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        seed=SEED,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    # Write minimal run metadata (nice for README)
    meta = {
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "train_size": len(ds["train"]),
        "val_size": len(ds["validation"]),
    }
    Path(OUT_DIR, "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("✅ Saved adapter to:", OUT_DIR)


if __name__ == "__main__":
    main()
