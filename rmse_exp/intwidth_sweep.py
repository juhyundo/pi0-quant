"""
rmse_exp/intwidth_sweep.py
---------------------------
Sweep IPT_INT_WIDTH_EXTRA from 0 to 15 (or a custom range) and measure RMSE
against a base server running bf16:bf16.

This script launches both servers itself:
  - base server   (bf16:bf16, default port 8000) — started once at the top
  - quantized server (ipt_c, default port 8002)  — restarted per sweep step
    with --int-width-extra X

Usage
-----
    python rmse_exp/intwidth_sweep.py \\
        --checkpoint-dir /path/to/checkpoint \\
        --input-fmt float8_e4m3 --output-fmt bfloat16 \\
        --n-obs 4 --seed 0 \\
        --min-extra 0 --max-extra 15 \\
        --gpu-base 0 --gpu-quant 1
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import IO, Optional

import numpy as np
import torch

_THIS_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_CLIENT_SRC = _REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"
for _p in [str(_REPO_ROOT), str(_CLIENT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openpi_client import websocket_client_policy as _ws

from rmse_exp.server_utils import (
    _kill_listeners_on_port,
    _open_step_log,
    _random_observation_droid,
    _stop_proc_tree,
    _timestamp_tag,
    _to_actions_tensor,
    _wait_for_port,
    _wait_until_ready,
    _with_fixed_pi0_noise,
)

logger = logging.getLogger(__name__)

_SERVE_SCRIPT = _REPO_ROOT / "pi0_inout_c" / "serve_quant.py"


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def _rmse(base: list[torch.Tensor], quantized: list[torch.Tensor]) -> float:
    r = torch.cat(base)
    n = torch.cat(quantized)
    return math.sqrt(float((r - n).pow(2).mean().item()))


# ---------------------------------------------------------------------------
# Server launchers
# ---------------------------------------------------------------------------

def _start_base_server(
    *,
    python: str,
    checkpoint_dir: str,
    config: str,
    port: int,
    gpu: int,
    openpi_dir: Optional[str],
    log_path: Path,
) -> subprocess.Popen:
    cmd = [
        python, str(_SERVE_SCRIPT),
        "--config",         config,
        "--checkpoint-dir", checkpoint_dir,
        "--port",           str(port),
        "--gpu",            "0",
        "--input-fmt",      "bfloat16",
        "--output-fmt",     "bfloat16",
    ]
    if openpi_dir:
        cmd += ["--openpi-dir", openpi_dir]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w")
    logger.info("Starting base server: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )


def _start_quantized_server(
    *,
    python: str,
    checkpoint_dir: str,
    config: str,
    port: int,
    gpu: int,
    input_fmt: str,
    output_fmt: str,
    int_width_extra: int,
    seed: int,
    openpi_dir: Optional[str],
    stdout: Optional[IO[str]] = None,
) -> subprocess.Popen:
    cmd = [
        python, str(_SERVE_SCRIPT),
        "--config",           config,
        "--checkpoint-dir",   checkpoint_dir,
        "--port",             str(port),
        "--gpu",              "0",
        "--input-fmt",        input_fmt,
        "--output-fmt",       output_fmt,
        "--int-width-extra",  str(int_width_extra),
        "--seed",             str(seed),
    ]
    if openpi_dir:
        cmd += ["--openpi-dir", openpi_dir]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    return subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        force=True,
    )

    p = argparse.ArgumentParser(
        description="Sweep IPT_INT_WIDTH_EXTRA (0..15) and measure RMSE vs base bf16:bf16 server"
    )
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--config", default="pi05_droid_jointpos_polaris")
    p.add_argument("--openpi-dir", default=None)
    p.add_argument("--python", default=None,
                   help="Python executable to use (default: same as this process)")

    p.add_argument("--base-port",      type=int, default=8000)
    p.add_argument("--quantized-port", type=int, default=8002)
    p.add_argument("--gpu-base",       type=int, default=0)
    p.add_argument("--gpu-quant",      type=int, default=1)

    p.add_argument("--input-fmt",  default="float8_e4m3",
                   help="input_fmt for the quantized server (default: float8_e4m3)")
    p.add_argument("--output-fmt", default="bfloat16",
                   help="output_fmt for the quantized server (default: bfloat16)")

    p.add_argument("--min-extra", type=int, default=0,
                   help="Start of int_width_extra sweep inclusive (default: 0)")
    p.add_argument("--max-extra", type=int, default=15,
                   help="End of int_width_extra sweep inclusive (default: 15)")

    p.add_argument("--n-obs", type=int, default=1)
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--no-fixed-pi0-noise", action="store_true",
                   help="Disable fixed diffusion noise injection (not recommended: "
                        "RMSE will include sampling variance)")
    p.add_argument("--ready-timeout-s", type=float, default=120.0)
    p.add_argument("--log-dir", default=str(_REPO_ROOT / "rmse_exp" / "logs"))
    args = p.parse_args()

    use_fixed_pi0_noise = not args.no_fixed_pi0_noise

    python = args.python or sys.executable
    openpi_dir = args.openpi_dir or str(_REPO_ROOT / "openpi")

    log_root = Path(args.log_dir)
    if not log_root.is_absolute():
        log_root = _REPO_ROOT / log_root
    run_dir = (log_root / f"intwidth-{_timestamp_tag()}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = (run_dir / "run.log").open("w", encoding="utf-8")

    rng = np.random.default_rng(args.seed)

    # ── Start base server (bf16:bf16) ────────────────────────────────────────
    _kill_listeners_on_port(args.base_port)
    _kill_listeners_on_port(args.quantized_port)

    base_proc = _start_base_server(
        python=python,
        checkpoint_dir=args.checkpoint_dir,
        config=args.config,
        port=args.base_port,
        gpu=args.gpu_base,
        openpi_dir=openpi_dir,
        log_path=run_dir / "base_server.log",
    )
    logger.info("Waiting for base server on port %d...", args.base_port)
    if not _wait_for_port(args.base_port, timeout_s=args.ready_timeout_s):
        logger.error("Base server did not start within %.1fs — aborting", args.ready_timeout_s)
        _stop_proc_tree(base_proc)
        sys.exit(1)
    logger.info("Base server ready on port %d", args.base_port)

    base = _ws.WebsocketClientPolicy(host="127.0.0.1", port=args.base_port)

    # Warm-up call to determine action shape and action_dim from metadata
    obs0 = _random_observation_droid(rng)
    _wait_until_ready(base, obs0, timeout_s=args.ready_timeout_s)
    warm = base.infer(obs0)
    action_horizon = np.asarray(warm["actions"]).shape[0]
    base_md = base.get_server_metadata() or {}
    action_dim = int(base_md.get("action_dim", 32))

    # Build fixed observations for the sweep
    observations: list[dict] = []
    for _ in range(args.n_obs):
        obs = _random_observation_droid(rng)
        if use_fixed_pi0_noise:
            obs = _with_fixed_pi0_noise(
                obs, rng=rng, action_horizon=action_horizon, action_dim=action_dim
            )
        observations.append(obs)

    # Collect base actions once — reused for every sweep step
    logger.info("Collecting base actions (%d observations)...", args.n_obs)
    base_actions: list[torch.Tensor] = [
        _to_actions_tensor(base.infer(obs)) for obs in observations
    ]

    run_log.write(f"# run_dir={run_dir}\n")
    run_log.write(f"# base port={args.base_port}  quantized port={args.quantized_port}\n")
    run_log.write(f"# n_obs={args.n_obs}  seed={args.seed}\n")
    run_log.write(f"# sweep: int_width_extra {args.min_extra}..{args.max_extra}\n")
    run_log.write(f"# input_fmt={args.input_fmt}  output_fmt={args.output_fmt}\n")
    run_log.write(f"# gpu_base={args.gpu_base}  gpu_quant={args.gpu_quant}\n")
    run_log.write(f"# use_fixed_pi0_noise={use_fixed_pi0_noise}  action_dim={action_dim}\n\n")
    run_log.flush()

    quantized_proc: Optional[subprocess.Popen] = None
    quantized_log_fh: Optional[IO[str]] = None

    try:
        for extra in range(args.min_extra, args.max_extra + 1):
            # ── (Re)start quantized server with new int_width_extra ───────────
            if quantized_proc is not None and quantized_proc.poll() is None:
                _stop_proc_tree(quantized_proc)
            _kill_listeners_on_port(args.quantized_port)
            if quantized_log_fh is not None:
                quantized_log_fh.close()

            step_tag = f"int_width_extra={extra:02d}"
            quantized_log_fh = _open_step_log(log_dir=run_dir, tag=step_tag)
            quantized_log_fh.write(f"# {step_tag}\n")
            quantized_log_fh.write(
                f"# input_fmt={args.input_fmt}  output_fmt={args.output_fmt}\n\n"
            )
            quantized_log_fh.flush()

            logger.info("[%s] Starting quantized server...", step_tag)
            quantized_proc = _start_quantized_server(
                python=python,
                checkpoint_dir=args.checkpoint_dir,
                config=args.config,
                port=args.quantized_port,
                gpu=args.gpu_quant,
                input_fmt=args.input_fmt,
                output_fmt=args.output_fmt,
                int_width_extra=extra,
                seed=args.seed,
                openpi_dir=openpi_dir,
                stdout=quantized_log_fh,
            )

            if not _wait_for_port(args.quantized_port, timeout_s=args.ready_timeout_s):
                logger.error(
                    "[%s] Quantized server did not start within %.1fs — skipping",
                    step_tag, args.ready_timeout_s,
                )
                line = f"{step_tag:25s}  rmse=nan  (server failed to start)"
                print(line)
                run_log.write(line + "\n")
                run_log.flush()
                continue

            quantized = _ws.WebsocketClientPolicy(host="127.0.0.1", port=args.quantized_port)
            _wait_until_ready(quantized, obs0, timeout_s=args.ready_timeout_s)

            # ── Query quantized server ────────────────────────────────────────
            quant_actions: list[torch.Tensor] = [
                _to_actions_tensor(quantized.infer(obs)) for obs in observations
            ]

            rmse = _rmse(base_actions, quant_actions)
            line = f"{step_tag:25s}  rmse={rmse:.4e}"

            print(line)
            run_log.write(line + "\n")
            run_log.flush()
            quantized_log_fh.write(f"\n# result: {line}\n")
            quantized_log_fh.flush()

            if not math.isfinite(rmse):
                logger.warning("[%s] Non-finite RMSE", step_tag)

    finally:
        logger.info("Stopping base server...")
        _stop_proc_tree(base_proc)
        if quantized_proc is not None and quantized_proc.poll() is None:
            _stop_proc_tree(quantized_proc)
        run_log.close()
        if quantized_log_fh is not None:
            quantized_log_fh.close()

    print(f"\nDone. Logs in: {run_dir}")


if __name__ == "__main__":
    main()
