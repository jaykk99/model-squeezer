"""Quantization logic for model-squeezer.

Uses llama.cpp's `quantize` binary for GGUF files. For safetensors/bin
files we attempt an HF->GGUF conversion first when the tools are available,
otherwise we report honestly that the file cannot be quantized.
"""
import os
import re
import shutil
import subprocess

# Jay-facing level name -> llama.cpp quant type
QUANT_LEVELS = {
    "Q8": "Q8_0",
    "Q6": "Q6_K",
    "Q5": "Q5_K_M",
    "Q4": "Q4_K_M",
    "Q3": "Q3_K_M",
    "Q2": "Q2_K",
    "Q1": "IQ1_M",   # extreme 1-bit-ish, Chromebook mode
}

# Approx bits per parameter for each quant type (weights only).
# Used for honest size/RAM estimates before running the real quantize.
BITS_PER_PARAM = {
    "Q8_0": 8.5,
    "Q6_K": 6.6,
    "Q5_K_M": 5.5,
    "Q4_K_M": 4.9,
    "Q3_K_M": 3.9,
    "Q2_K": 3.4,
    "IQ1_M": 1.8,
    "F16": 16.0,
    "F32": 32.0,
}

CHROMEBOOK_RAM_MB = 4096


def find_quantize_bin():
    """Locate the llama.cpp quantize binary."""
    env = os.environ.get("LLAMA_QUANTIZE_BIN")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [
        os.path.join(here, "..", "bin", "quantize"),
        os.path.join(here, "bin", "quantize"),
        "/usr/local/bin/quantize",
    ]:
        if os.path.isfile(cand):
            return cand
    found = shutil.which("quantize")
    if found:
        return found
    return None


def gguf_param_count(path):
    """Read parameter count from GGUF metadata. Returns int or None."""
    try:
        from gguf import GGUFReader
        reader = GGUFReader(path)
        for field in reader.fields.values():
            pass
        # tensor element counts
        total = 0
        for tensor in reader.tensors:
            n = 1
            for d in tensor.shape:
                n *= int(d)
            total += n
        return total if total > 0 else None
    except Exception:
        return None


def estimate_size_mb(params, quant_type):
    """Honest size estimate: params * bits/8."""
    bits = BITS_PER_PARAM.get(quant_type)
    if not bits or not params:
        return None
    return params * bits / 8 / 1024 / 1024


def run_quantize(in_path, out_path, quant_type, on_log=None):
    """Run llama.cpp quantize. Returns (ok, message)."""
    bin_path = find_quantize_bin()
    if not bin_path:
        return False, "quantize binary not found. The server needs llama.cpp built."
    if not os.path.isfile(in_path):
        return False, "input file not found: %s" % in_path
    cmd = [bin_path, in_path, out_path, quant_type]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=6 * 3600
        )
    except subprocess.TimeoutExpired:
        return False, "quantize timed out after 6 hours."
    except Exception as e:
        return False, "quantize failed to start: %s" % e
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if on_log:
        on_log(log[-2000:])
    if proc.returncode != 0 or not os.path.isfile(out_path):
        return False, "quantize failed: %s" % log[-500:].strip()
    return True, "done"


def chromebook_plan(params):
    """Build the honest Chromebook-mode plan for a param count.

    Returns list of dicts: level, est_mb, fits_4gb. Also a verdict string.
    We never claim a 70B model fits in 4GB: the math is shown plainly.
    """
    plan = []
    for level, qtype in QUANT_LEVELS.items():
        mb = estimate_size_mb(params, qtype)
        if mb is None:
            continue
        plan.append({
            "level": level,
            "quant_type": qtype,
            "est_mb": round(mb, 1),
            "fits_4gb": mb <= CHROMEBOOK_RAM_MB,
        })
    smallest = min(plan, key=lambda r: r["est_mb"]) if plan else None
    # recommend the BEST quality level that still fits (first in Q8..Q1 order)
    best_fit = next((r for r in plan if r["fits_4gb"]), None)
    if best_fit:
        verdict = "Fits in 4GB at %s (about %s MB). Best quality that fits." % (
            best_fit["level"], best_fit["est_mb"])
    else:
        verdict = (
            "This model cannot fit in 4GB RAM even at maximum compression "
            "(smallest is about %s MB at %s). Pick a smaller base model: "
            "a 7B model at Q4 is about 4GB, a 3B model at Q4 is about 2GB."
            % (smallest["est_mb"], smallest["level"]) if smallest else "unknown size."
        )
    return plan, verdict


def fmt_mb(mb):
    if mb is None:
        return "unknown"
    if mb >= 1024:
        return "%.2f GB" % (mb / 1024)
    return "%.1f MB" % mb
