# Model Squeezer

## Live URL

https://model-squeezer.onrender.com

Compress any AI model as many times as you want. Less RAM, same model. Shows the size before and after every squeeze.

## What it does

**Compress tab**
- Get a model from 4 sources: file upload, GitHub repo, Hugging Face, or any direct URL
- Pick a compression level: Q8, Q6, Q5, Q4, Q3, Q2, Q1 (extreme)
- Hit "Compress again" to chain levels: Q8 then Q4 then Q2
- See before vs after size, MB and percent saved, estimated RAM, and a download link

**Chromebook Mode**
- Goal: run huge models on weak hardware (4GB RAM, CPU only, no GPU)
- Shows honest math for every level: estimated size and whether it fits in 4GB
- No fake promises. A 70B model cannot fit in 4GB at any level, and the table proves it.
  A 7B model at Q4 is about 4GB. A 3B model at Q4 is about 2GB.

**Train tab**
- Fine-tune a model on CPU with LoRA. No GPU needed.
- Two methods: standard LoRA, or LoRA Low-Memory (Adafactor plus gradient
  checkpointing for weak CPUs)
- Upload .txt or .jsonl training data, set epochs and learning rate,
  watch the loss curve and ETA, download the trained adapter

**History tab**
- Every squeeze ever run, with before/after sizes and download links

## How it works

GGUF models are quantized with llama.cpp's `quantize` binary (built in the
Docker image). Training uses PyTorch CPU plus PEFT LoRA. Storage is local
disk, so files disappear on redeploy (Render free tier).

## Run locally

```
pip install -r requirements.txt
uvicorn app.main:app --app-dir . --port 8000
```

You also need the llama.cpp `quantize` binary, or set `LLAMA_QUANTIZE_BIN`
to its path.
