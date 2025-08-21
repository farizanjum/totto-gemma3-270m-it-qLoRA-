import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

RESPONSE_TAG = "### OUTPUT:\n"
SYSTEM = (
    "You are a data-to-text model. "
    "Given a table and highlighted cells, write ONE faithful sentence."
)
PROMPT = (
    SYSTEM + "\n### TABLE\n{table}\n" + RESPONSE_TAG
)

OUT_DIR = "outputs/totto-gemma3-270m-it-lora"
BASE_MODEL = "google/gemma-3-270m-it"


def main():
    ex = load_dataset("json", data_files={"val": "data/processed/val.jsonl"})["val"][0]

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model then apply LoRA adapter
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, OUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(OUT_DIR, use_fast=False)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    user_content = PROMPT.format(table=ex["text"])  # single-turn user prompt
    chat = [{"role": "user", "content": user_content}]
    inputs = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs.shape[1]
    out = model.generate(
        input_ids=inputs,
        max_new_tokens=64,
        min_new_tokens=5,
        do_sample=False,  # greedy decoding
        num_beams=1,
    )
    new_tokens = out[0][input_len:]
    print("GEN:", tokenizer.decode(new_tokens, skip_special_tokens=True).split("\n")[0].strip())
    print("REF:", ex["labels"])


if __name__ == "__main__":
    main()
