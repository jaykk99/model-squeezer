import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from quant import (  # noqa: E402
    QUANT_LEVELS,
    QUANT_TYPE_TO_LEVEL,
    BITS_PER_PARAM,
    normalize_level,
    estimate_size_mb,
    size_estimates_mb,
    fmt_mb,
    chromebook_plan,
    find_quantize_bin,
    gguf_param_count,
)


def test_quant_levels_complete():
    assert set(QUANT_LEVELS) == {"Q8", "Q6", "Q5", "Q4", "Q3", "Q2", "Q1"}
    assert QUANT_LEVELS["Q4"] == "Q4_K_M"
    assert QUANT_TYPE_TO_LEVEL["Q4_K_M"] == "Q4"


def test_normalize_level_canonical():
    assert normalize_level("Q4") == "Q4"
    assert normalize_level("q4") == "Q4"
    assert normalize_level("  q4  ") == "Q4"
    assert normalize_level("Q1") == "Q1"


def test_normalize_level_raw_types():
    assert normalize_level("Q4_K_M") == "Q4"
    assert normalize_level("q4_k_m") == "Q4"
    assert normalize_level("Q8_0") == "Q8"
    assert normalize_level("IQ1_M") == "Q1"
    assert normalize_level("iq1_m") == "Q1"


def test_normalize_level_rejects_garbage():
    assert normalize_level("bogus") is None
    assert normalize_level("") is None
    assert normalize_level(None) is None
    assert normalize_level(42) is None
    assert normalize_level("Q9") is None


def test_estimate_size_mb_math():
    # 7B params at Q4_K_M (4.9 bits): 7e9 * 4.9 / 8 bytes -> ~4096 MB
    mb = estimate_size_mb(7_000_000_000, "Q4_K_M")
    assert mb is not None
    assert 3900 < mb < 4300


def test_estimate_size_mb_bad_input():
    assert estimate_size_mb(0, "Q4_K_M") is None
    assert estimate_size_mb(None, "Q4_K_M") is None
    assert estimate_size_mb(7_000_000_000, "NOPE") is None


def test_fmt_mb():
    assert fmt_mb(2048) == "2.00 GB"
    assert fmt_mb(512) == "512.0 MB"
    assert fmt_mb(None) == "unknown"


def test_chromebook_plan_7b_fits():
    plan, verdict = chromebook_plan(7_000_000_000)
    assert len(plan) == 7
    q4 = next(r for r in plan if r["level"] == "Q4")
    assert q4["fits_4gb"] is True
    assert "Fits in 4GB" in verdict


def test_chromebook_plan_70b_honest():
    plan, verdict = chromebook_plan(70_000_000_000)
    assert all(not r["fits_4gb"] for r in plan)
    assert "cannot fit in 4GB" in verdict
    # plan stays in Q8..Q1 order (quality first)
    assert [r["level"] for r in plan] == list(QUANT_LEVELS)


def test_size_estimates_mb():
    ests = size_estimates_mb(7_000_000_000)
    assert len(ests) == 7
    assert ests[0]["level"] == "Q8"
    assert all(e["est_mb"] > 0 for e in ests)
    # deeper compression saves more vs the F16 baseline
    savings = [e["saved_pct_vs_f16"] for e in ests]
    assert savings == sorted(savings)
    assert ests[-1]["level"] == "Q1"
    assert ests[-1]["saved_pct_vs_f16"] > 80


def test_size_estimates_mb_bad_input():
    assert size_estimates_mb(0) == []
    assert size_estimates_mb(None) == []
    assert size_estimates_mb(-5) == []


def test_find_quantize_bin_env_override(tmp_path):
    fake = tmp_path / "quantize"
    fake.write_bytes(b"#!/bin/sh\n")
    old = os.environ.get("LLAMA_QUANTIZE_BIN")
    os.environ["LLAMA_QUANTIZE_BIN"] = str(fake)
    try:
        assert find_quantize_bin() == str(fake)
    finally:
        if old is None:
            del os.environ["LLAMA_QUANTIZE_BIN"]
        else:
            os.environ["LLAMA_QUANTIZE_BIN"] = old


def test_gguf_param_count_missing_file():
    assert gguf_param_count("/does/not/exist.gguf") is None


def test_bits_per_param_covers_all_levels():
    for qtype in QUANT_LEVELS.values():
        assert qtype in BITS_PER_PARAM
