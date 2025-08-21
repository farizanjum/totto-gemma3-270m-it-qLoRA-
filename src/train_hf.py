import os
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import yaml
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def yload(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SYSTEM = (
    "You are a data-to-text model. "
    "Given a table and highlighted cells, write ONE faithful sentence."
)
RESPONSE_TAG = "### OUTPUT:\n"


def format_example(example: Dict) -> Dict:
    text = example["text"]
    label = example["labels"]
    prompt = f"{SYSTEM}\n### TABLE\n{text}\n{RESPONSE_TAG}{label}"
    return {"text": prompt}


def find_sublist(haystack: List[int], needle: List[int]) -> int:
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    for i in range(0, n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return -1


def build_tokenize_and_mask(tokenizer: AutoTokenizer, max_len: int):
    resp_ids = tokenizer(RESPONSE_TAG, add_special_tokens=False)["input_ids"]

    def _fn(example: Dict) -> Dict:
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        idx = find_sublist(input_ids, resp_ids)
        # Labels = input_ids, but mask everything before RESPONSE_TAG
        labels = list(input_ids)
        if idx != -1:
            tag_end = idx + len(resp_ids)
            for i in range(0, tag_end):
                labels[i] = -100
        else:
            # If not found, mask all tokens to avoid training on prompt accidentally
            labels = [-100] * len(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }

    return _fn


def main(cfg_path: str = "configs/totto_it_270m.yaml") -> None:
    cfg = yload(cfg_path)
    seed_all(int(cfg.get("seed", 42)))

    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Load processed data
    train_ds = load_dataset("json", data_files={"train": "data/processed/train.jsonl"})["train"]
    val_ds = load_dataset("json", data_files={"val": "data/processed/val.jsonl"})["val"]

    # Subsample for fast loop if requested
    subsample = int(cfg.get("subsample_train", 0) or 0)
    if subsample and subsample < len(train_ds):
        train_ds = train_ds.shuffle(seed=42).select(range(subsample))

    # Format examples into single "text" field with RESPONSE_TAG masking point
    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names, num_proc=1)
    # Keep val untouched (used only for BLEU later)

    # Tokenizer
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_len = int(cfg.get("max_seq_len", 1024))
    tok_fn = build_tokenize_and_mask(tokenizer, max_len)

    train_tok = train_ds.map(tok_fn, remove_columns=["text"], num_proc=1)

    # Quantization + PEFT
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=False,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora = cfg["lora"]
    peft_cfg = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    # Training args (no in-loop eval to avoid signature mismatch issues)
    trn = cfg["train"]
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=float(trn["epochs"]),
        per_device_train_batch_size=int(trn["per_device_train_bs"]),
        gradient_accumulation_steps=int(trn["grad_accum_steps"]),
        learning_rate=float(trn["lr"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(trn["warmup_ratio"]),
        logging_steps=int(trn["logging_steps"]),
        save_strategy=str(trn["save_strategy"]),
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,
        dataloader_num_workers=0,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
