#!/usr/bin/env bash
set -e
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# Install CUDA wheels first (much faster + stable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Rest of stack (prefer wheels)
pip install --prefer-binary --no-cache-dir \
  transformers>=4.43 datasets>=2.20 accelerate>=0.33 trl>=0.9.6 \
  evaluate sacrebleu>=2.4 pandas numpy tqdm sentencepiece protobuf \
  peft>=0.11 bitsandbytes>=0.43 unsloth
echo "Now run: huggingface-cli login  (accept Gemma license)"
