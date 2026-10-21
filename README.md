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

## API extras

- `GET /api/stats` — model count, disk used, history count, quantize availability, active jobs
- `GET /api/models/{id}` — one model
- `GET /api/jobs?limit=20` — recent jobs, newest first (log tails stripped; fetch one job for full detail)
- `GET /api/models/{id}/estimates` — honest per-level size estimates for a model with known params
- `DELETE /api/models/{id}` — delete a model and its file; refuses (409) when other entries were derived from it, so chained compressions never lose their source
- `DELETE /api/history` — clear compression history
- `GET /api/compress/preview?model_id=&level=` — honest before/after prediction for one squeeze (needs known params)
- `GET /api/models/{id}/chain` — the compress-again chain this model belongs to, oldest first
- `PATCH /api/models/{id}` `{"name": "…"}` — rename a model (display name only)
- `GET /api/history/export?fmt=csv` — download history as CSV (default JSON)
- Compression accepts the level in any case (`q4`, `Q4`) or the raw llama.cpp type (`Q4_K_M`)
- `/api/train` validates its inputs server-side (epochs 1-20, lr (0,1], batch 1-8, grad_accum 1-64, seq len 16-4096, training data non-empty and under 200 MB)
- `download_url` only allows http(s) URLs — `file://`, `ftp://`, etc. are rejected so a crafted URL can never make the server touch local files

## Tests

```
pip install -r requirements.txt   # plus pytest
python -m pytest tests/ -q
```

44 tests cover quant math and Chromebook verdicts, source helpers and URL-scheme
rejection, and the API (stats, jobs, level normalization, model delete with
child protection, train validation, error paths) against a throwaway data dir.

## Run locally

```
pip install -r requirements.txt
uvicorn app.main:app --app-dir . --port 8000
```

You also need the llama.cpp `quantize` binary, or set `LLAMA_QUANTIZE_BIN`
to its path.
