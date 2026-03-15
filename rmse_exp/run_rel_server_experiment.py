"""
run_rel_server_experiment.py
-----------------------------
Single-shot RMSE measurement on two live policy servers using relative-error noise.

Mirrors run_ulp_server_experiment.py but for rel_err-based noise injection.
The --rel-err / --input-fmt / --output-fmt flags are client-side labels only —
they describe what the servers were started with, not control them.

Usage
-----
  1) Start base server on port 8000 (--rel-err 0)
  2) Start quantized server on port 8002 (--rel-err <value>)
  3) Run this script:

    python rmse_exp/run_rel_server_experiment.py \\
        --base-port 8000 --quantized-port 8002 \\
        --rel-err 1e-3 --input-fmt bfloat16 --output-fmt bfloat16 \\
        --n-obs 8

Prints:
    rmse=<value>  base[...]  quantized[...]
    THRESHOLD VIOLATED    (if rmse >= --rmse-threshold)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OPENPI_CLIENT_SRC = _REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"
for _p in [str(_REPO_ROOT), str(_OPENPI_CLIENT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openpi_client import websocket_client_policy as _ws


def _random_observation_droid(rng: np.random.Generator) -> dict:
    return {
        "observation/exterior_image_1_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left":      rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position":        rng.random(7, dtype=np.float32),
        "observation/gripper_position":      rng.random(1, dtype=np.float32),
        "prompt": "do something",
    }


def _to_actions_tensor(resp: dict) -> torch.Tensor:
    a = resp["actions"]
    t = torch.from_numpy(np.asarray(a).copy()).float()
    return t.reshape(-1)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Single-shot RMSE between two live policy servers (rel_err variant)."
    )
    p.add_argument("--base-host",      default="127.0.0.1")
    p.add_argument("--base-port",      type=int, default=8000)
    p.add_argument("--quantized-host", default="127.0.0.1")
    p.add_argument("--quantized-port", type=int, default=8002)
    p.add_argument("--n-obs",  type=int,   default=1)
    p.add_argument("--seed",   type=int,   default=0)
    p.add_argument("--rmse-threshold", type=float, default=0.4)

    # Label-only flags — describe what was passed to the servers, don't control them.
    p.add_argument("--base-rel-err",    type=float, default=None)
    p.add_argument("--base-input-fmt",  default=None)
    p.add_argument("--base-output-fmt", default=None)
    p.add_argument("--rel-err",         type=float, default=None,
                   help="rel_err the quantized server was started with (label only)")
    p.add_argument("--input-fmt",       default=None)
    p.add_argument("--output-fmt",      default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    base      = _ws.WebsocketClientPolicy(host=args.base_host,      port=args.base_port)
    quantized = _ws.WebsocketClientPolicy(host=args.quantized_host, port=args.quantized_port)

    # Warmup + determine action shape.
    obs0 = _random_observation_droid(rng)
    warm = base.infer(obs0)
    quantized.infer(obs0)
    action_horizon = np.asarray(warm["actions"]).shape[0]
    base_md        = base.get_server_metadata() or {}
    action_dim     = int(base_md.get("action_dim", 32))

    base_actions      = []
    quantized_actions = []

    for _ in range(args.n_obs):
        obs = _random_observation_droid(rng)
        # Deterministic diffusion noise so identical servers produce identical outputs.
        obs["pi0_noise"] = rng.standard_normal((action_horizon, action_dim)).astype(np.float32)
        base_actions.append(_to_actions_tensor(base.infer(obs)))
        quantized_actions.append(_to_actions_tensor(quantized.infer(obs)))

    b    = torch.cat(base_actions,      dim=0)
    q    = torch.cat(quantized_actions, dim=0)
    rmse = math.sqrt(float((b - q).pow(2).mean().item()))

    base_label = (
        f"rel_err={args.base_rel_err} in={args.base_input_fmt} out={args.base_output_fmt}"
    )
    quant_label = (
        f"rel_err={args.rel_err} in={args.input_fmt} out={args.output_fmt}"
    )
    print(f"rmse={rmse:.4e}  base[{base_label}]  quantized[{quant_label}]")
    if rmse >= args.rmse_threshold:
        print("THRESHOLD VIOLATED")


if __name__ == "__main__":
    main()
