"""
rmse_exp/intwidth_sweep_inproc.py
----------------------------------
Like intwidth_sweep.py but loads the base bf16:bf16 model in-process instead
of launching a separate base server.  This eliminates the ~70s base-server
startup overhead.

  - Base model: loaded in-process on --gpu-base, no WebSocket round-trip.
  - Quantized server: still launched as a subprocess per sweep step (needs
    --int-width-extra reload each time).

Usage
-----
    python rmse_exp/intwidth_sweep_inproc.py \\
        --checkpoint-dir /path/to/checkpoint \\
        --config pi0_droid \\
        --gpu-base 2 \\
        --quantized-port 8003 --gpu-quant 3 \\
        --input-fmt float8_e4m3 --output-fmt bfloat16 \\
        --min-extra 0 --max-extra 15
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
_OPENPI_SRC = _REPO_ROOT / "openpi" / "src"
_CLIENT_SRC = _REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"
for _p in [str(_REPO_ROOT), str(_OPENPI_SRC), str(_CLIENT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openpi_client import websocket_client_policy as _ws

from pi0_inout_c.serve_quant import (
    Pi0PyTorchPolicy,
    _get_model_config,
    _load_norm_stats,
    load_pi0_pytorch,
)
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
# Quantized server launcher
# ---------------------------------------------------------------------------

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
    norm_stats_dir: Optional[str] = None,
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
    if norm_stats_dir:
        cmd += ["--norm-stats-dir", norm_stats_dir]

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
        description=(
            "Sweep IPT_INT_WIDTH_EXTRA (0..15) and measure RMSE vs in-process "
            "bf16:bf16 base model (no base server)"
        )
    )
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--config", default="pi05_droid_jointpos_polaris")
    p.add_argument("--openpi-dir", default=None)
    p.add_argument("--python", default=None,
                   help="Python executable for the quantized server subprocess "
                        "(default: same as this process)")

    p.add_argument("--gpu-base",       type=int, default=0,
                   help="GPU index for the in-process base model")
    p.add_argument("--quantized-port", type=int, default=8002)
    p.add_argument("--gpu-quant",      type=int, default=1)

    p.add_argument("--input-fmt",  default="float8_e4m3")
    p.add_argument("--output-fmt", default="bfloat16")

    p.add_argument("--min-extra", type=int, default=0)
    p.add_argument("--max-extra", type=int, default=15)

    p.add_argument("--n-obs", type=int, default=1)
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--no-fixed-pi0-noise", action="store_true")
    p.add_argument("--ready-timeout-s", type=float, default=120.0)
    p.add_argument("--norm-stats-dir", default=None)
    p.add_argument("--log-dir", default=str(_REPO_ROOT / "rmse_exp" / "logs"))
    args = p.parse_args()

    use_fixed_pi0_noise = not args.no_fixed_pi0_noise
    python = args.python or sys.executable
    openpi_dir = args.openpi_dir or str(_REPO_ROOT / "openpi")

    log_root = Path(args.log_dir)
    if not log_root.is_absolute():
        log_root = _REPO_ROOT / log_root
    run_dir = (log_root / f"intwidth-inproc-{_timestamp_tag()}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = (run_dir / "run.log").open("w", encoding="utf-8")

    rng = np.random.default_rng(args.seed)

    # ── Load base model in-process ───────────────────────────────────────────
    base_device = torch.device(f"cuda:{args.gpu_base}")
    logger.info("Loading base model in-process on %s ...", base_device)
    base_model = load_pi0_pytorch(args.config, args.checkpoint_dir, base_device)

    # Norm stats: explicit dir > checkpoint_dir/assets/droid fallback
    norm_stats = None
    if args.norm_stats_dir:
        norm_stats = _load_norm_stats(args.norm_stats_dir)
        logger.info("Loaded norm stats from %s", args.norm_stats_dir)
    else:
        fallback = Path(args.checkpoint_dir) / "assets" / "droid"
        if (fallback / "norm_stats.json").exists():
            norm_stats = _load_norm_stats(str(fallback))
            logger.info("Loaded norm stats from %s", fallback)
        else:
            logger.warning(
                "No --norm-stats-dir and no norm_stats.json in %s — "
                "running WITHOUT normalization", fallback
            )

    cfg = _get_model_config(args.config)
    use_quantile_norm = getattr(cfg, "pi05", False)
    is_joint_position = "jointpos" in args.config

    base_policy = Pi0PyTorchPolicy(
        model=base_model,
        device=base_device,
        norm_stats=norm_stats,
        use_quantile_norm=use_quantile_norm,
        is_joint_position=is_joint_position,
        max_token_len=cfg.max_token_len,
    )
    logger.info("Base model ready on %s", base_device)

    # ── Build observations ───────────────────────────────────────────────────
    action_horizon = cfg.action_horizon
    action_dim     = cfg.action_dim

    # warmup obs for quantized server readiness checks
    obs0 = _random_observation_droid(rng)

    observations: list[dict] = []
    for _ in range(args.n_obs):
        obs = _random_observation_droid(rng)
        if use_fixed_pi0_noise:
            obs = _with_fixed_pi0_noise(
                obs, rng=rng, action_horizon=action_horizon, action_dim=action_dim
            )
        observations.append(obs)

    # ── Collect base actions once ────────────────────────────────────────────
    logger.info("Collecting base actions (%d observations)...", args.n_obs)
    base_actions: list[torch.Tensor] = []
    for obs in observations:
        base_actions.append(_to_actions_tensor(base_policy.infer(obs)))

    run_log.write(f"# run_dir={run_dir}\n")
    run_log.write(f"# base=in-process gpu={args.gpu_base}  quantized port={args.quantized_port}\n")
    run_log.write(f"# n_obs={args.n_obs}  seed={args.seed}\n")
    run_log.write(f"# sweep: int_width_extra {args.min_extra}..{args.max_extra}\n")
    run_log.write(f"# input_fmt={args.input_fmt}  output_fmt={args.output_fmt}\n")
    run_log.write(f"# gpu_base={args.gpu_base}  gpu_quant={args.gpu_quant}\n")
    run_log.write(f"# use_fixed_pi0_noise={use_fixed_pi0_noise}  action_dim={action_dim}\n\n")
    run_log.flush()

    _kill_listeners_on_port(args.quantized_port)

    quantized_proc: Optional[subprocess.Popen] = None
    quantized_log_fh: Optional[IO[str]] = None

    try:
        for extra in range(args.min_extra, args.max_extra + 1):
            # ── (Re)start quantized server with new int_width_extra ──────────
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
                norm_stats_dir=args.norm_stats_dir,
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

            # ── Query quantized server ───────────────────────────────────────
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
        if quantized_proc is not None and quantized_proc.poll() is None:
            _stop_proc_tree(quantized_proc)
        run_log.close()
        if quantized_log_fh is not None:
            quantized_log_fh.close()

    print(f"\nDone. Logs in: {run_dir}")


if __name__ == "__main__":
    main()
