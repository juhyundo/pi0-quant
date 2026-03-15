"""
quant_linear_c.py
---------------
QuantLinearC: a drop-in replacement for nn.Linear that applies quantization
to matmul inputs (activation + weight) and matmul output separately.

RTL-backed variant
------------------
This version replaces F.linear(...) with CIPTLinearRTLFunction so the core
matmul/bias path uses the *C-accelerated* model of the RTL MXU.

The C backend is compiled on first import (one-time gcc call) and cached as
a module-level singleton shared across all QuantLinear instances.  Set the
environment variable IPT_HEADER_DIR to the directory containing the C headers
before importing this module.

Important limitations
---------------------
The current RTL model is hardcoded for:
  - input_fmt  = E4M3   (activations / weights / bias)
  - psumFmt    = BF16
  - outputFmt  = BF16 container, with optional runtime OutBF16 / OutE4M3

So this wrapper only supports configurations that match the RTL:
  - input_fmt  must be E4M3
  - output_fmt must be BF16 or E4M3

Forward pass semantics
----------------------
Given input_fmt A and output_fmt B:

    x_q      = quant(x,    A)              # activation loaded from memory in A
    W_q      = quant(W,    A)              # weight loaded from memory in A
    b_q      = quant(bias, A)              # bias loaded from memory in A
    y_accum  = c_rtl_linear(x_q, W_q, b_q) # tiled E4M3xE4M3 -> BF16/E4M3 via C model
    result   = quant(y_accum, B)           # optional final write-to-memory quantization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .quant_types import QuantFormat, quant
from .stats_tracker import StatsTracker, Component
from .ipt_mxu_model.ipt_rtl_linear_c import CIPTLinearRTLFunction
from .ipt_mxu_model.fp_formats import OutputFmtSel
from .rel_noise import RelNoiseConfig, inject_rel_noise


_SUPPORTED_E4M3_NAMES = {
    "e4m3",
    "fp8_e4m3",
    "float8_e4m3",
    "ocp_e4m3",
}

_SUPPORTED_BF16_NAMES = {
    "bf16",
    "bfloat16",
}


class QuantLinearC(nn.Module):
    """
    Drop-in replacement for nn.Linear with configurable input/output quantization.

    This RTL-backed version uses CIPTLinearRTLFunction (C-accelerated) instead
    of the Python IPTLinearRTLFunction for the quantized matmul path.

    The public interface is identical to the Python-backed version.

    Attributes:
        weight:       Shared with original linear (not copied).
        bias:         Shared with original linear (not copied).
        input_fmt:    Applied to both activation x and weight W before matmul.
        output_fmt:   Applied to final output writeback.
        component:    Architectural component tag (vision/language/action_*).
        layer_name:   Full dot-separated module path, used as stats key.
        tracker:      Optional StatsTracker for RMSE collection.
    """

    def __init__(
        self,
        linear: nn.Linear,
        input_fmt: QuantFormat,
        output_fmt: QuantFormat,
        component: Component,
        layer_name: str,
        tracker: Optional[StatsTracker] = None,
        noise_cfg: Optional[RelNoiseConfig] = None,
        *,
        vec_len: int = 32,
        num_lanes: int = 16,
        pipeline_depth: int = 1,
        scale_exp: int = 0,
        int_width_extra: int = 15,
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

        self.input_fmt = input_fmt
        self.output_fmt = output_fmt
        self.component = component
        self.layer_name = layer_name
        self.tracker = tracker
        self.noise_cfg = noise_cfg

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.vec_len = vec_len
        self.num_lanes = num_lanes
        self.pipeline_depth = pipeline_depth
        self.scale_exp = scale_exp
        self.int_width_extra = int_width_extra

        in_name = self._fmt_name(input_fmt)
        out_name = self._fmt_name(output_fmt)
        self._use_rtl = (
            in_name in _SUPPORTED_E4M3_NAMES
            and out_name in (_SUPPORTED_E4M3_NAMES | _SUPPORTED_BF16_NAMES)
        )

        if self._use_rtl:
            self.out_fmt_sel = self._to_output_fmt_sel(output_fmt)
            # Constructing CIPTLinearRTLFunction triggers the one-time gcc
            # compilation (if not already done) and validates the headers are
            # reachable.  Subsequent instances reuse the cached .so.
            self.rtl_linear = CIPTLinearRTLFunction(
                vec_len=vec_len,
                num_lanes=num_lanes,
                pipeline_depth=pipeline_depth,
                out_fmt_sel=self.out_fmt_sel,
                int_width_extra=int_width_extra,
            )
            # Pre-quantize and cache weights on CPU — weights are static at
            # inference time so there is no need to re-quantize every forward.
            self._w_q_cpu = quant(linear.weight.cpu().float(), input_fmt)
            self._b_q_cpu = (
                quant(linear.bias.float(), input_fmt).cpu()
                if linear.bias is not None else None
            )
        else:
            self.out_fmt_sel = None
            self.rtl_linear = None
            self._w_q_cpu = None
            self._b_q_cpu = None

    @staticmethod
    def _fmt_name(fmt: QuantFormat) -> str:
        return str(fmt.value).lower()

    @staticmethod
    def _to_output_fmt_sel(fmt: QuantFormat) -> OutputFmtSel:
        name = str(fmt.value).lower()
        if name in _SUPPORTED_BF16_NAMES:
            return OutputFmtSel.OutBF16
        if name in _SUPPORTED_E4M3_NAMES:
            return OutputFmtSel.OutE4M3
        raise ValueError(f"Unsupported RTL output format: {fmt.value!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        # All reference arithmetic starts from float32 tensors.
        x_f32 = x.float()
        w_f32 = self.weight.float()
        b_f32 = self.bias.float() if self.bias is not None else None

        # ── Load all inputs from memory in input_fmt ─────────────────────────
        x_q = quant(x_f32, self.input_fmt)
        # w_q and b_q are pre-quantized at init for the RTL path; fall back to
        # on-the-fly quantization only for the F.linear path.
        if self._use_rtl:
            w_q = self._w_q_cpu
            b_q = self._b_q_cpu
        else:
            w_q = quant(w_f32, self.input_fmt)
            b_q = quant(b_f32, self.input_fmt) if b_f32 is not None else None

        # ── Accumulate using C RTL model (or F.linear fallback) ─────────────
        if self._use_rtl:
            y_accum = self.rtl_linear(
                x_q.cpu(),
                w_q,
                b_q,
                scale_exp=self.scale_exp,
            ).to(x_f32.device)
        else:
            y_accum = F.linear(x_q, w_q, b_q)

        # ── Optional relative-error noise injection ──────────────────────────
        if self.noise_cfg is not None and self.noise_cfg.enabled():
            y_accum = inject_rel_noise(y_accum, rel_err=self.noise_cfg.rel_err)

        # ── Write result to output memory in output_fmt ──────────────────────
        y_out = quant(y_accum, self.output_fmt)

        # ── RMSE: compare against unquantized full-precision reference ───────
        if self.tracker is not None:
            with torch.no_grad():
                y_fp = F.linear(x_f32, w_f32, b_f32)
                self.tracker.record(
                    name=self.layer_name,
                    component=self.component,
                    y_fp=y_fp,
                    y_quant=y_out,
                )

        return y_out.to(original_dtype)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"input_fmt={self.input_fmt.value}, output_fmt={self.output_fmt.value}, "
            f"component={self.component.value}, rtl={self._use_rtl}, "
            f"vec_len={self.vec_len}, num_lanes={self.num_lanes}, "
            f"pipeline_depth={self.pipeline_depth}, scale_exp={self.scale_exp}, "
            f"int_width_extra={self.int_width_extra}"
        )
