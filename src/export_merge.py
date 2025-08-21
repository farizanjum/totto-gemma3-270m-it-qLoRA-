import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "google/gemma-3-270m-it"
ADAPTER_DIR = "outputs/totto-gemma3-270m-it-lora"
OUT_DIR = "outputs/totto-gemma3-270m-it-merged-fp16"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load base model in FP16 on CPU to keep VRAM low during merge
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        trust_remote_code=False,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    # Merge LoRA weights into the base and unload PEFT wrappers
    merged = model.merge_and_unload()  # returns base model with weights merged

    # Save merged model and tokenizer
    merged.save_pretrained(OUT_DIR, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=False)
    tok.save_pretrained(OUT_DIR)

    print(f"Merged FP16 model saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
