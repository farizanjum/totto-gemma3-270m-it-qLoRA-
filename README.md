# ToTTo → Gemma‑3‑270M‑IT (4‑bit QLoRA, Windows‑friendly)

Fine‑tuning Google Gemma‑3‑270M‑IT on the ToTTo dataset using vanilla Transformers + PEFT (LoRA) with 4‑bit quantization. Optimized to run on an RTX 4050 (6 GB) on Windows without Unsloth or Triton headaches.

## TL;DR
- Base model: `google/gemma-3-270m-it`
- Method: 4‑bit QLoRA + LoRA (r=8, α=16, dropout=0.05)
- Hardware: single RTX 4050 (6 GB), Windows 10/11
- Dataset: `GEM/totto` (table → sentence)
- Time: ~49 min for 500 steps (1 epoch on 8k subsample)
- Final train loss: `1.455` (reported by Trainer)
- Adapter size: `~7.6 MB` (`outputs/totto-gemma3-270m-it-lora/adapter_model.safetensors`)
- Merged fp16 model size: `~536 MB` (`outputs/totto-gemma3-270m-it-merged-fp16/model.safetensors`)

> Heads‑up: Deterministic BLEU with greedy decoding is usually conservative for single‑sentence data‑to‑text; evaluate with PARENT/BLEURT or add light sampling for better correlation with human judgments.

---

## Results (first run)

- Training config (summarized):
  - batch size: 1; grad accum: 16; epochs: 1; lr: 5e‑5; warmup: 0.06; cosine schedule
  - sequence length: 1024; 4‑bit quant; LoRA target modules: q/k/v/o, up/down/gate
- Trainer logs (subset):
  - step 25:  loss 2.698
  - step 50:  loss 2.020
  - step 100: loss 1.534
  - step 250: loss 1.325
  - step 500: loss 1.327 (epoch end)
  - final reported train loss: **1.455**
- Wallclock: **~49 min** (500 steps; ~0.17 steps/s; ~2.72 samples/s)
- Model sizes:
  - LoRA adapter: **7.6 MB** (trainable deltas only)
  - Merged fp16: **536 MB** (for deployment without PEFT)

> BLEU (greedy, 100 val) on the Windows stack can be pessimistic; use `src/eval_hf.py` (merged model) or `src/eval_hf.py` (base+adapter) and consider non‑greedy decoding for qualitative checks.

---

## Project structure
```
.
├─ configs/
│  └─ totto_it_270m.yaml
├─ data/
│  └─ processed/                # generated JSONL lives here
├─ outputs/
│  ├─ totto-gemma3-270m-it-lora/        # LoRA adapter (7.6 MB)
│  └─ totto-gemma3-270m-it-merged-fp16/ # merged fp16 (536 MB)
├─ scripts/
│  └─ 01_env.sh                 # linux/mac venv bootstrap
└─ src/
   ├─ preprocess.py             # ToTTo → {text, labels}
   ├─ train_hf.py               # Transformers + PEFT 4‑bit QLoRA
   ├─ eval_hf.py                # BLEU (greedy) – merged fp16 by default
   ├─ infer_hf.py               # quick generation (base + adapter)
   └─ export_merge.py           # merge adapter → fp16 single checkpoint
```

---

## Setup

Windows PowerShell (recommended):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Minimal deps
pip install transformers peft datasets accelerate bitsandbytes sacrebleu sentencepiece protobuf
# Optional: gh CLI for GitHub push
```

Linux/mac (bash):
```bash
bash scripts/01_env.sh
```

Login for gated base model:
```powershell
.venv\Scripts\huggingface-cli.exe login
```

---

## Data prep
```powershell
.venv\Scripts\python.exe src\preprocess.py
# writes: data/processed/train.jsonl, val.jsonl
```

---

## Train (4‑bit QLoRA, PEFT LoRA)
```powershell
.venv\Scripts\python.exe src\train_hf.py
# adapter saved under: outputs/totto-gemma3-270m-it-lora
```

---

## Export (merge adapter → fp16)
```powershell
.venv\Scripts\python.exe src\export_merge.py
# merged fp16 saved under: outputs/totto-gemma3-270m-it-merged-fp16
```

---

## Evaluate (BLEU; greedy)
Merged fp16 path (default):
```powershell
.venv\Scripts\python.exe src\eval_hf.py
# prints: BLEU: xx.xx
```

For adapter path evaluation (alternative): switch to the adapter loader variant if desired.

---

## Inference
Adapter path (base + LoRA):
```powershell
.venv\Scripts\python.exe src\infer_hf.py
```

Merged fp16 (alternative):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("outputs/totto-gemma3-270m-it-merged-fp16", torch_dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained("outputs/totto-gemma3-270m-it-merged-fp16")
```

---

## Why this is Windows‑friendly
- No Unsloth/Triton patching
- No datasets multiprocessing (single‑proc map)
- TorchDynamo/Inductor disabled during eval to avoid Triton kernel compile errors
- 4‑bit quantization keeps VRAM ~6 GB safe

---

## Notes & licenses
- Base model: `google/gemma-3-270m-it` (accept license on Hugging Face).
- Dataset: `GEM/totto`.
- This repo saves **adapters** by default; do not push large weights to GitHub (see `.gitignore`). Use `export_merge.py` for a single fp16 deployable checkpoint.

---

## Bib & credits
- Gemma‑3: Google
- Transformers/PEFT/Accelerate: Hugging Face
- ToTTo dataset: Google Research
