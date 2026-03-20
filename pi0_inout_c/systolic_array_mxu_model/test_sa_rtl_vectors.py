"""
test_sa_rtl_vectors.py
======================
Feeds inputs from mxu_full_matmul_vectors.txt through the C systolic array
model and compares against the RTL simulation outputs in
rtl_sa_full_matmul_outputs.txt.

  mxu_full_matmul_vectors.txt   -- inputs (wgt, act) + Python-golden exp lines
  rtl_sa_full_matmul_outputs.txt -- actual RTL simulation outputs

The C model output is compared against the RTL output (actual_hex column),
NOT the Python golden (expected_hex column).

rtl_sa_full_matmul_outputs.txt format (from SystolicArrayFullMatmulTest.scala):
  # case_id  tile  row  col  actual_hex  expected_hex  ...
  0    7    0    0   0xc5f2   0xc5fa   ...

  The file has one header comment line then data rows.
  We only use: case_id, row, col, actual_hex.
  tile = num_tiles-1 (the final tile output) -- we only care about the last tile.

Run:
    python3 -m pi0_inout_c.systolic_array_mxu_model.test_sa_rtl_vectors \\
        --vectors /path/to/mxu_full_matmul_vectors.txt \\
        --rtl-outputs /path/to/rtl_sa_full_matmul_outputs.txt \\
        -v
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# SA geometry
# ---------------------------------------------------------------------------

_ROWS = 32
_COLS = 16
_SA_USE_PSUM = 1
_SA_USE_ZERO = 2

# ---------------------------------------------------------------------------
# C shim
# ---------------------------------------------------------------------------

_SHIM_C = textwrap.dedent(
    """\
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "fp_formats.h"
#include "converters.h"
#include "systolic_array_model.h"

void sa_test_init(void) {
    atlas_fp_init_lut();
    atlas_acc_init_lut();
}

SAModel *sa_test_new(int rows, int cols) {
    SystolicArrayParams p;
    p.rows = rows;
    p.cols = cols;
    return sa_model_init(&p);
}

void sa_test_free(SAModel *m)  { sa_model_free(m); }
void sa_test_reset(SAModel *m) { sa_model_reset(m); }

int sa_test_load_weights(SAModel *m,
                         int col_idx, int buf_write_sel,
                         const uint8_t *weights, int n)
{
    SA_WeightLoadReq req;
    req.col_idx              = col_idx;
    req.weight_buf_write_sel = buf_write_sel;
    req.weights              = weights;
    req.weights_len          = n;
    return sa_load_weights(m, &req) ? 0 : -1;
}

void sa_test_compute(SAModel *m,
                     const uint8_t  *act,
                     const uint16_t *psum,
                     int             addend_sel,
                     uint16_t       *out)
{
    uint8_t bias_zero[64] = {0};
    SA_ComputeReq cr;
    cr.activation_row      = act;
    cr.bias                = bias_zero;
    cr.psum                = psum;
    cr.addend_sel          = (SA_AddendSel)addend_sel;
    cr.weight_buf_read_sel = false;
    cr.scale_exp           = 0;
    cr.out_fmt_sel         = OutputFmtSel_OutBF16;
    sa_compute_now(m, &cr, out);
}
"""
)

_lib: Optional[ctypes.CDLL] = None
_build_dir: Optional[str] = None


def _get_lib() -> ctypes.CDLL:
    global _lib, _build_dir
    if _lib is not None:
        return _lib

    header_dir = os.environ.get(
        "SA_HEADER_DIR",
        os.path.dirname(os.path.abspath(__file__)),
    )
    _build_dir = tempfile.mkdtemp(prefix="sa_test_")
    shim_c = os.path.join(_build_dir, "sa_shim.c")
    shim_so = os.path.join(_build_dir, "libsa_test.so")

    with open(shim_c, "w") as f:
        f.write(_SHIM_C)

    cmd = [
        "gcc",
        "-O2",
        "-Wall",
        "-Wno-unused-function",
        "-shared",
        "-fPIC",
        f"-I{header_dir}",
        "-o",
        shim_so,
        shim_c,
        "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        shutil.rmtree(_build_dir, ignore_errors=True)
        raise RuntimeError(
            f"C shim compile failed:\n{r.stderr}\n" f"SA_HEADER_DIR={header_dir!r}"
        )

    lib = ctypes.CDLL(shim_so)
    lib.sa_test_init.restype = None
    lib.sa_test_init.argtypes = []
    lib.sa_test_new.restype = ctypes.c_void_p
    lib.sa_test_new.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.sa_test_free.restype = None
    lib.sa_test_free.argtypes = [ctypes.c_void_p]
    lib.sa_test_reset.restype = None
    lib.sa_test_reset.argtypes = [ctypes.c_void_p]
    lib.sa_test_load_weights.restype = ctypes.c_int
    lib.sa_test_load_weights.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
    ]
    lib.sa_test_compute.restype = None
    lib.sa_test_compute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.sa_test_init()
    _lib = lib
    return _lib


class CSAModel:
    def __init__(self):
        self._lib = _get_lib()
        self._m = self._lib.sa_test_new(_ROWS, _COLS)
        assert self._m

    def __del__(self):
        if self._m:
            self._lib.sa_test_free(self._m)
            self._m = None

    def reset(self):
        self._lib.sa_test_reset(self._m)

    def load_weights(self, col: int, buf: int, weights: list[int]):
        w = (ctypes.c_uint8 * len(weights))(*weights)
        rc = self._lib.sa_test_load_weights(
            self._m, ctypes.c_int(col), ctypes.c_int(buf), w, ctypes.c_int(len(weights))
        )
        assert rc == 0

    def compute(self, act: list[int], psum: list[int], addend_sel: int) -> list[int]:
        a = (ctypes.c_uint8 * len(act))(*act)
        ps = (ctypes.c_uint16 * len(psum))(*psum)
        out = (ctypes.c_uint16 * _COLS)()
        self._lib.sa_test_compute(self._m, a, ps, ctypes.c_int(addend_sel), out)
        return list(out)


# ---------------------------------------------------------------------------
# Parse mxu_full_matmul_vectors.txt  (inputs only)
# ---------------------------------------------------------------------------


@dataclass
class MatmulVector:
    id: int
    num_rows: int = 64
    num_tiles: int = 1
    weights: list[list[list[int]]] = field(default_factory=list)  # [tile][col][row]
    acts: list[list[list[int]]] = field(default_factory=list)  # [tile][row][k]


def _parse_inputs(path: str) -> list[MatmulVector]:
    vectors: list[MatmulVector] = []
    cur_id: Optional[int] = None
    num_rows = 64
    num_tiles = 1
    wgt_map: dict[int, dict[int, list[int]]] = {}
    act_map: dict[int, dict[int, list[int]]] = {}

    def flush():
        nonlocal cur_id, num_rows, num_tiles, wgt_map, act_map
        if cur_id is None:
            return
        weights = [
            [wgt_map.get(t, {}).get(c, [0] * _ROWS) for c in range(_COLS)]
            for t in range(num_tiles)
        ]
        acts = [
            [act_map.get(t, {}).get(r, [0] * _ROWS) for r in range(num_rows)]
            for t in range(num_tiles)
        ]
        vectors.append(
            MatmulVector(
                id=cur_id,
                num_rows=num_rows,
                num_tiles=num_tiles,
                weights=weights,
                acts=acts,
            )
        )
        cur_id = None
        num_rows = 64
        num_tiles = 1
        wgt_map.clear()
        act_map.clear()

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                flush()
                continue
            if line.startswith("#"):
                flush()
                cur_id = int(line[1:].split()[0])
                continue
            p = line.split()
            if p[0] == "num_rows":
                num_rows = int(p[1])
            elif p[0] == "num_tiles":
                num_tiles = int(p[1])
            elif p[0] == "wgt":
                t, c = int(p[1]), int(p[2])
                wgt_map.setdefault(t, {})[c] = [int(x, 16) for x in p[3:]]
            elif p[0] == "act":
                t, r = int(p[1]), int(p[2])
                act_map.setdefault(t, {})[r] = [int(x, 16) for x in p[3:]]
            # skip exp lines — we use the RTL output file instead
    flush()
    return vectors


# ---------------------------------------------------------------------------
# Parse rtl_sa_full_matmul_outputs.txt  (RTL actual outputs)
#
# Format:
#   # case_id  tile  row  col  actual_hex  expected_hex  ...
#   0    7    0    0   0xc5f2   0xc5fa   ...
#
# We load the LAST tile for each (case_id, row, col) since that's the final
# accumulated output after all tiles.
# ---------------------------------------------------------------------------

# rtl_outputs[case_id][row][col] -> int (BF16 bits from actual_hex)
RTLOutputs = dict[int, dict[int, dict[int, int]]]


def _parse_rtl_outputs(path: str) -> RTLOutputs:
    out: RTLOutputs = {}
    # Track max tile seen per case so we only keep the last tile's row
    max_tile: dict[int, int] = {}

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                case_id = int(parts[0])
                tile = int(parts[1])
                row = int(parts[2])
                col = int(parts[3])
                actual_hex = parts[4]
                actual = int(actual_hex, 16)
            except (ValueError, IndexError):
                continue

            # Only keep the highest tile index (= final accumulated output)
            prev_max = max_tile.get(case_id, -1)
            if tile < prev_max:
                continue
            if tile > prev_max:
                # New max tile: discard earlier tile data for this case
                max_tile[case_id] = tile
                if case_id in out:
                    out[case_id].clear()

            out.setdefault(case_id, {}).setdefault(row, {})[col] = actual

    return out


# ---------------------------------------------------------------------------
# Run one vector through the C model
# ---------------------------------------------------------------------------


def _run_vector(tv: MatmulVector, dut: CSAModel) -> list[list[int]]:
    dut.reset()
    psum = [[0] * _COLS for _ in range(tv.num_rows)]
    results = [[0] * _COLS for _ in range(tv.num_rows)]

    for t in range(tv.num_tiles):
        for col in range(_COLS):
            dut.load_weights(col, 0, tv.weights[t][col])

        addend_sel = _SA_USE_ZERO if t == 0 else _SA_USE_PSUM

        for row_idx in range(tv.num_rows):
            out = dut.compute(tv.acts[t][row_idx], psum[row_idx], addend_sel)
            results[row_idx] = out
            psum[row_idx] = out

    return results


# ---------------------------------------------------------------------------
# BF16 helper
# ---------------------------------------------------------------------------


def _bf16f(bits: int) -> float:
    return struct.unpack("f", struct.pack("I", (bits & 0xFFFF) << 16))[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_tests(vectors_path: str, rtl_path: str, verbose: bool) -> bool:
    for p, label in [(vectors_path, "vectors"), (rtl_path, "rtl-outputs")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} file not found: {p}")
            return False

    print("Compiling C shim ...")
    dut = CSAModel()

    print(f"Loading inputs  from {vectors_path}")
    inputs = _parse_inputs(vectors_path)
    print(f"Loading RTL outputs from {rtl_path}")
    rtl_out = _parse_rtl_outputs(rtl_path)
    print(f"Loaded {len(inputs)} test vector(s).\n")

    total_pass = total_fail = 0

    for tv in inputs:
        if tv.id not in rtl_out:
            print(f"Case {tv.id:3d}: SKIP (no RTL output found)")
            continue

        results = _run_vector(tv, dut)
        rtl_rows = rtl_out[tv.id]

        case_pass = case_fail = 0
        nonzero_c = nonzero_rtl = 0
        for row in range(tv.num_rows):
            for col in range(_COLS):
                c_val = results[row][col] & 0xFFFF
                rtl_val = rtl_rows.get(row, {}).get(col, None)
                if rtl_val is None:
                    continue
                rtl_val &= 0xFFFF

                if c_val != 0:
                    nonzero_c += 1
                if rtl_val != 0:
                    nonzero_rtl += 1

                if c_val == rtl_val:
                    case_pass += 1
                else:
                    case_fail += 1
                    if verbose:
                        ulp = abs(c_val - rtl_val)
                        print(
                            f"  FAIL case={tv.id} row={row:2d} col={col:2d}"
                            f"  C=0x{c_val:04x}({_bf16f(c_val):.5g})"
                            f"  RTL=0x{rtl_val:04x}({_bf16f(rtl_val):.5g})"
                            f"  ulp={ulp}"
                        )

        total_cells = tv.num_rows * _COLS
        total_pass += case_pass
        total_fail += case_fail
        status = "PASS" if case_fail == 0 else "FAIL"
        print(
            f"Case {tv.id:3d} [{tv.num_rows} rows x {tv.num_tiles} tiles]: "
            f"{status}  {case_pass}/{total_cells} correct"
            f"  (nonzero: C={nonzero_c} RTL={nonzero_rtl})"
            + (f"  ({case_fail} failures)" if case_fail else "")
        )

    grand = total_pass + total_fail
    print(
        f"\nTotal: {total_pass}/{grand} cells matched RTL"
        + (f"  ({total_fail} mismatches)" if total_fail else "  -- ALL MATCH")
    )
    return total_fail == 0


def main() -> None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _res = os.path.join(_here, "..", "..", "src", "test", "resources")

    parser = argparse.ArgumentParser(
        description="Compare C SA model against RTL simulation outputs"
    )
    parser.add_argument(
        "--vectors",
        default=os.path.join(_res, "mxu_full_matmul_vectors.txt"),
        help="Path to mxu_full_matmul_vectors.txt (inputs)",
    )
    parser.add_argument(
        "--rtl-outputs",
        default=os.path.join(_res, "rtl_sa_full_matmul_outputs.txt"),
        help="Path to rtl_sa_full_matmul_outputs.txt (RTL actuals)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print every failing cell"
    )
    args = parser.parse_args()

    ok = run_tests(args.vectors, args.rtl_outputs, args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
