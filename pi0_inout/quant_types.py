"""
quant_types.py
--------------
Quantization format definitions for the five formats under test:
FLOAT32 (baseline), FP16, BF16, FP8-E4M3, FP8-E5M2.

Quantization is performed using PyTorch's native dtype casting:

    x_quantized = x.float().to(target_dtype).to(x.dtype)

The intermediate cast to float32 avoids compounding precision loss when the
input is already in a reduced format.  The cast to target_dtype rounds each
value to the nearest representable number in that format using IEEE 754
round-to-nearest-even.  The final cast back restores the original storage
dtype so downstream ops see the same type they expect.

FLOAT32 is a special passthrough: x.float().to(float32).to(x.dtype) is
bitwise identical to x.  RMSE for a FLOAT32/FLOAT32 run is exactly 0.

FP8 notes:
    torch.float8_e4m3fn  — 1 sign, 4 exponent, 3 mantissa bits.
                           Finite range ±448.  NaN represented; no ±Inf.
    torch.float8_e5m2    — 1 sign, 5 exponent, 2 mantissa bits.
                           Finite range ±57344.  Has NaN and ±Inf.
    Both require PyTorch >= 2.1.
"""

from __future__ import annotations

import torch
from enum import Enum


# ---------------------------------------------------------------------------
# Format enum
# ---------------------------------------------------------------------------

class QuantFormat(str, Enum):
    FLOAT32     = "float32"      # passthrough — no rounding, zero RMSE baseline
    FLOAT16     = "float16"
    BFLOAT16    = "bfloat16"
    FLOAT8_E4M3 = "float8_e4m3"  # torch.float8_e4m3fn
    FLOAT8_E5M2 = "float8_e5m2"  # torch.float8_e5m2


# Map QuantFormat → torch.dtype
TORCH_DTYPE: dict[QuantFormat, torch.dtype] = {
    QuantFormat.FLOAT32:     torch.float32,
    QuantFormat.FLOAT16:     torch.float16,
    QuantFormat.BFLOAT16:    torch.bfloat16,
    QuantFormat.FLOAT8_E4M3: torch.float8_e4m3fn,
    QuantFormat.FLOAT8_E5M2: torch.float8_e5m2,
}

# Format metadata for reporting
FORMAT_BITS: dict[QuantFormat, dict] = {
    QuantFormat.FLOAT32:     {"total": 32, "exp": 8,  "mantissa": 23},
    QuantFormat.FLOAT16:     {"total": 16, "exp": 5,  "mantissa": 10},
    QuantFormat.BFLOAT16:    {"total": 16, "exp": 8,  "mantissa": 7},
    QuantFormat.FLOAT8_E4M3: {"total": 8,  "exp": 4,  "mantissa": 3},
    QuantFormat.FLOAT8_E5M2: {"total": 8,  "exp": 5,  "mantissa": 2},
}


# ---------------------------------------------------------------------------
# Core quantization function
# ---------------------------------------------------------------------------

_FP8_FORMATS = {QuantFormat.FLOAT8_E4M3, QuantFormat.FLOAT8_E5M2}

# Max representable finite value for each FP8 format
_FP8_MAX: dict[QuantFormat, float] = {
    QuantFormat.FLOAT8_E4M3: 448.0,
    QuantFormat.FLOAT8_E5M2: 57344.0,
}


def quant(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    Quantize tensor `x` to format `fmt` and return in the original dtype.

    For FLOAT32: returns x unchanged (identity — no rounding).
    For FP8 formats: applies per-tensor absmax scaling to fit the dynamic
        range into the representable range, then casts and scales back.
        This prevents overflow → NaN while preserving relative precision.
    For FP16/BF16: raw cast (range is sufficient for typical values).

    Args:
        x:   Input tensor in any floating-point dtype.
        fmt: Target format.

    Returns:
        Tensor of same shape and dtype as `x`.
    """
    target = TORCH_DTYPE[fmt]

    if fmt in _FP8_FORMATS:
        return _quant_fp8_scaled(x, fmt)

    # x.float() → target_dtype (RNE rounding) → original_dtype
    # For FLOAT32, this is float32 → float32 → original = x  (identity)
    return x.float().to(target).to(x.dtype)


def _quant_fp8_scaled(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    Per-tensor absmax scaling for FP8 quantization.

    scale = max(|x|) / fp8_max
    x_scaled = x / scale          → fits within [-fp8_max, fp8_max]
    x_q = cast(x_scaled, fp8)     → round to FP8 grid
    return x_q * scale            → restore original magnitude
    """
    target = TORCH_DTYPE[fmt]
    fp8_max = _FP8_MAX[fmt]

    x_f32 = x.float()
    amax = x_f32.abs().max()

    # If all zeros, nothing to scale
    if amax == 0:
        return x_f32.to(x.dtype)

    scale = amax / fp8_max
    x_scaled = x_f32 / scale
    x_q = x_scaled.to(target).to(torch.float32)
    return (x_q * scale).to(x.dtype)


def all_formats() -> list[QuantFormat]:
    return list(QuantFormat)


def sweep_pairs(
    include_baseline: bool = True,
) -> list[tuple[QuantFormat, QuantFormat]]:
    """
    Return all (input_fmt, output_fmt) combinations.

    With include_baseline=True (default): 5×5 = 25 pairs including FLOAT32.
    With include_baseline=False: 4×4 = 16 pairs, reduced formats only.
    """
    fmts = all_formats() if include_baseline else [
        f for f in QuantFormat if f != QuantFormat.FLOAT32
    ]
    return [(inf, outf) for inf in fmts for outf in fmts]
