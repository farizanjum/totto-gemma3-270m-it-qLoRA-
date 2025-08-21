import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sacrebleu
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Improve stability/perf warnings
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

RESPONSE_TAG = "### OUTPUT:\n"
SYSTEM = (
    "You are a data-to-text model. "
    "Given a table and highlighted cells, write ONE faithful sentence."
)
PROMPT = (
    SYSTEM + "\n### TABLE\n{table}\n" + RESPONSE_TAG
)

MERGED_DIR = "outputs/totto-gemma3-270m-it-merged-fp16"


def clean_text(s: str) -> str:
    s = s.replace("<pad>", "").strip()
    return s


def main():
    val = load_dataset("json", data_files={"val": "data/processed/val.jsonl"})["val"]

    # Load merged FP16 model directly
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, use_fast=False)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    hyps, refs = [], []
    for ex in val.select(range(min(100, len(val)))):
        user_content = PROMPT.format(table=ex["text"])  # single-turn user prompt
        chat = [{"role": "user", "content": user_content}]
        inputs = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        input_len = inputs.shape[1]
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs,
                max_new_tokens=64,
                min_new_tokens=5,
                do_sample=False,
                num_beams=1,
            )
        new_tokens = out[0][input_len:]
        hyp = tokenizer.decode(new_tokens, skip_special_tokens=True)
        hyp = clean_text(hyp).split("\n")[0].strip()
        ref = clean_text(ex["labels"]).strip()
        hyps.append(hyp)
        refs.append(ref)

    print(f"BLEU: {sacrebleu.corpus_bleu(hyps, [refs]).score:.2f}")


if __name__ == "__main__":
    main()
