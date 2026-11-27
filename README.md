# Model Squeezer

## Live URL

Currently suspended on Render (no live host yet).

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
- Direct model downloads reject non-http(s) schemes, URL-embedded credentials, loopback/private/reserved destination IPs and redirects to those addresses. Downloads are capped at 10 GiB by default; set `SQ_MAX_DOWNLOAD_BYTES` to override the byte ceiling. Incomplete downloads are deleted after errors. This check is defense-in-depth; deploy with network-level egress restrictions to address DNS rebinding and infrastructure metadata risks.

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


## Adaptive local model runtime (Stretch + Model Squeezer)

The shared Python 3.9+ launcher uses llama.cpp for CPU inference. It does not
require an API key or a hosted server. Run from either repository checkout:

    python3 stretch_model.py owner/model-GGUF --plan
    python3 stretch_model.py owner/model-GGUF --prompt "Hello"
    python3 stretch_model.py /path/to/model.gguf --prompt "Hello"

On Windows use `python` instead of `python3`. For a public direct HTTPS GGUF URL,
pass the URL as the first argument. Use --sha256 with a trusted file digest
for integrity; pinned URLs are recommended.

The launcher detects available RAM, sets a conservative weight budget, selects a
single-file GGUF from a repository containing variants of the same model,
downloads in bounded chunks, reuses the local cache, and sets context and
thread counts. It uses an installed llama-cli or builds llama.cpp from source
on first run (git, CMake, and C++ compiler required). No GPU is required.
For Android/Termux install prerequisites:

    pkg install python git cmake clang make

For Debian/Ubuntu/Chromebook Linux:

    sudo apt install python3 git cmake build-essential

For Windows, install Python, Git, CMake and Visual Studio C++ Build Tools,
or an existing llama-cli from llama.cpp; for macOS install Python, Git, CMake
and Xcode command-line tools. Pass --runtime /path/to/llama-cli to reuse a
native runtime; --no-build refuses to download runtime source.

Both repositories use the same STRETCH_MODEL_CACHE directory (default
~/.cache/stretch-models). Set --cache to a real USB/SSD filesystem path to
share storage between runs. The ordinary filesystem permits memory mapping;
Stretch's virtual disk HTTP endpoint is NOT a drop-in mmap filesystem.
The launcher does not inject Stretch's userspace memory store into llama.cpp.

For local unquantized GGUF, add --quantize Q4_K_M to create a smaller copy
without altering the original. Quantization itself needs RAM and temporary
disk; prefer a prequantized file on weak devices. Lossy quantization changes
weights and can degrade accuracy. Repeated quantization cannot shrink forever.
LoRA adapters and safetensors require architecture-compatible GGUF conversion
or merging first; arbitrary model architectures are not supported.

--disk-mode explicitly permits a larger-than-budget mmap experiment; it is
not a guarantee of usable speed or successful loading. --context and --ram-mib
allow stricter settings. Model file size is not total memory use; the KV cache
and architecture matter. First-time builds/downloads need internet; local
GGUF plus installed runtime works offline. iOS, browser-only devices, and
microcontrollers are not supported by this command-line launcher.

Tests: python -m unittest test_stretch_model -v. CI also attempts a real
small-model generation on Linux. Cross-platform test coverage does not
substitute for a physical Android/Chromebook inference test.


### Low-memory execution and layer streaming

GGUF on a constrained device:

    python stretch_model.py /path/to/model.gguf --disk-mode --context 512 --prompt "Hello"

The launcher probes the installed llama.cpp's actual flags. With modern
versions it enables mmap and lazy tensor reads (only tensors supported by
the runtime), disables weight repacking and warmup, bounds inference batch
sizes, and disables additional prompt caching where available. Older runtimes
fall back to advertised mmap flags. A one-shot prompt retries once at reduced
context after a possible memory allocation failure; --no-retry disables that.
Interactive sessions are not replayed. This is OS page-cache management, not
a custom GGUF layer scheduler or a guaranteed RAM cap.

Optional true module-level disk offload for local safetensors models, on
systems with compatible PyTorch wheels (desktop Linux/macOS/Windows):

    python -m pip install torch transformers accelerate safetensors
    python stretch_stream.py /path/to/local-model --offload-dir /path/to/ssd --ram-mib 2048 --plan
    python stretch_stream.py /path/to/local-model --offload-dir /path/to/ssd --ram-mib 2048 --prompt "Hello"

This builds an empty model to infer a CPU/disk device map, loads safe tensor
checkpoints, and uses Accelerate's execution hooks to bring disk-offloaded
weights into memory on demand. Temporary offload files are cleaned after
normal completion. KV caching is disabled to reduce persistent memory.
A CPU layer and working buffers must still fit; no total RSS guarantee.
--require-offload refuses runs where all weights fit in RAM. Disk offload can
be very slow, especially without an SSD. No remote custom model code is
executed. The optional backend is not yet supported on Termux/iOS. It uses
ordinary safetensors, not GGUF or arbitrary model formats.

Run the real synthetic Transformer smoke test with:

    python smoke_stretch_stream.py
