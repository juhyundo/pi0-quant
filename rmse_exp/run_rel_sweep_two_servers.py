"""
run_rel_sweep_two_servers.py
----------------------------------------------------------------
Automate a relative-error noise sweep using two policy servers:

  - base server, already running (default port 8000)
  - quantized server, optionally (re)started per sweep step with --rel-err = start,start+step,...
    (default port 8002)

For each rel_err value:
  - query both servers on the same set of observations
  - compute action RMSE
  - stop when RMSE >= threshold (default 0.4)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OPENPI_CLIENT_SRC = _REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"
for _p in [str(_REPO_ROOT), str(_OPENPI_CLIENT_SRC)]:
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
    _wait_until_ready,
    _with_fixed_pi0_noise,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metrics:
    rmse: float


def _metrics(base: list[torch.Tensor], quantized: list[torch.Tensor]) -> Metrics:
    r = torch.cat(base, dim=0)
    n = torch.cat(quantized, dim=0)
    diff = (r - n).abs()
    rmse = math.sqrt(float(diff.pow(2).mean().item()))
    return Metrics(rmse=rmse)


def _quant_cfg(policy: _ws.WebsocketClientPolicy) -> dict[str, Any]:
    try:
        md = policy.get_server_metadata() or {}
    except Exception:
        return {}
    q = md.get("quant", {})
    return q if isinstance(q, dict) else {}


def _argv_has(argv: list[str], flag: str) -> bool:
    return flag in argv


def _argv_set_kv(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            argv[i + 1] = value
        else:
            argv.append(value)
        return
    argv.extend([flag, value])


def _replace_placeholders(template: str, values: dict[str, object]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _start_quantized_server(
    quantized_server_cmd_template: str,
    *,
    rel_err: float,
    quantized_port: int,
    defaults: dict[str, Any],
    stdout: Optional[IO[str]] = None,
) -> subprocess.Popen:
    cmd_str = _replace_placeholders(
        quantized_server_cmd_template,
        {
            "rel_err": rel_err,
            "quantized_port": quantized_port,
            "input_fmt": defaults.get("input_fmt"),
            "output_fmt": defaults.get("output_fmt"),
        },
    )
    argv = shlex.split(cmd_str)

    _argv_set_kv(argv, "--rel-err", str(rel_err))
    _argv_set_kv(argv, "--port", str(quantized_port))

    if defaults.get("input_fmt") and not _argv_has(argv, "--input-fmt"):
        _argv_set_kv(argv, "--input-fmt", str(defaults["input_fmt"]))
    if defaults.get("output_fmt") and not _argv_has(argv, "--output-fmt"):
        _argv_set_kv(argv, "--output-fmt", str(defaults["output_fmt"]))

    return subprocess.Popen(
        argv,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-host", default="127.0.0.1")
    p.add_argument("--base-port", type=int, default=8000)
    p.add_argument("--quantized-host", default="127.0.0.1")
    p.add_argument("--quantized-port", type=int, default=8002)

    p.add_argument("--n-obs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rmse-threshold", type=float, default=0.4)
    p.add_argument("--start-rel-err", type=float, default=1e-4)
    p.add_argument("--rel-err-step", type=float, default=1e-4)
    p.add_argument("--ready-timeout-s", type=float, default=60.0)
    p.add_argument("--use-fixed-pi0-noise", action="store_true")
    p.add_argument("--log-dir", default=str(_REPO_ROOT / "rmse_exp" / "logs"),
                   help="Directory for per-step log files.")

    p.add_argument("--quantized-server-cmd", default=None)
    p.add_argument(
        "--kill-existing-quantized-server",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    base = _ws.WebsocketClientPolicy(host=args.base_host, port=args.base_port)
    base_q = _quant_cfg(base)

    start_rel_err = max(0.0, float(args.start_rel_err))
    rel_err_step  = max(1e-10, float(args.rel_err_step))

    log_root = Path(args.log_dir)
    if not log_root.is_absolute():
        log_root = _REPO_ROOT / log_root
    run_dir = (log_root / f"run-{_timestamp_tag()}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = (run_dir / "run.log").open("w", encoding="utf-8")

    obs0 = _random_observation_droid(rng)
    _wait_until_ready(base, obs0, timeout_s=args.ready_timeout_s)
    warm = base.infer(obs0)
    action_horizon = np.asarray(warm["actions"]).shape[0]
    action_dim = int(base_q.get("action_dim", 32))

    use_fixed_pi0_noise = args.use_fixed_pi0_noise

    run_log.write(f"# run_dir={run_dir}\n")
    run_log.write(f"# base={args.base_host}:{args.base_port}  quantized={args.quantized_host}:{args.quantized_port}\n")
    run_log.write(f"# n_obs={args.n_obs}  seed={args.seed}\n")
    run_log.write(f"# sweep: start_rel_err={start_rel_err:.4e}  rel_err_step={rel_err_step:.4e}\n")
    run_log.write(f"# use_fixed_pi0_noise={bool(use_fixed_pi0_noise)}\n")
    run_log.write(f"# quantized_server_cmd={args.quantized_server_cmd}\n\n")
    if base_q:
        run_log.write(f"# base_quant={base_q}\n\n")
    run_log.flush()

    observations: list[dict] = []
    for _ in range(args.n_obs):
        obs = _random_observation_droid(rng)
        if use_fixed_pi0_noise:
            obs = _with_fixed_pi0_noise(obs, rng=rng, action_horizon=action_horizon, action_dim=action_dim)
        observations.append(obs)

    def _sweep_values():
        i = 0
        while True:
            yield start_rel_err + i * rel_err_step
            i += 1

    quantized_proc: Optional[subprocess.Popen] = None
    quantized_log_fh: Optional[IO[str]] = None
    consecutive_nan = 0
    try:
        for rel_err in (_sweep_values() if args.quantized_server_cmd is not None else [None]):
            if args.quantized_server_cmd is None:
                rel_err = None
            else:
                if quantized_proc is not None and quantized_proc.poll() is None:
                    _stop_proc_tree(quantized_proc)
                if args.kill_existing_quantized_server:
                    _kill_listeners_on_port(args.quantized_port)
                if quantized_log_fh is not None:
                    quantized_log_fh.close()
                    quantized_log_fh = None

                step_tag = f"rel_err={rel_err:.4e}"
                quantized_log_fh = _open_step_log(log_dir=run_dir, tag=step_tag)
                quantized_log_fh.write(f"# {step_tag}\n")
                quantized_log_fh.write(f"# base={args.base_host}:{args.base_port}  quantized={args.quantized_host}:{args.quantized_port}\n")
                quantized_log_fh.write(f"# n_obs={args.n_obs}  seed={args.seed}\n")
                quantized_log_fh.write(f"# use_fixed_pi0_noise={bool(use_fixed_pi0_noise)}\n")
                quantized_log_fh.write(f"# quantized_server_cmd={args.quantized_server_cmd}\n\n")
                if base_q:
                    quantized_log_fh.write(f"# base_quant={base_q}\n\n")
                quantized_log_fh.flush()

                defaults = {
                    "input_fmt":  base_q.get("input_fmt", "bfloat16"),
                    "output_fmt": base_q.get("output_fmt", "bfloat16"),
                }
                quantized_proc = _start_quantized_server(
                    args.quantized_server_cmd,
                    rel_err=rel_err,
                    quantized_port=args.quantized_port,
                    defaults=defaults,
                    stdout=quantized_log_fh,
                )

            if args.quantized_server_cmd is None and quantized_log_fh is None:
                step_tag = "rel_err=as-is"
                quantized_log_fh = _open_step_log(log_dir=run_dir, tag=step_tag)
                quantized_log_fh.write(f"# {step_tag}\n")
                quantized_log_fh.write(f"# base={args.base_host}:{args.base_port}  quantized={args.quantized_host}:{args.quantized_port}\n")
                quantized_log_fh.write(f"# n_obs={args.n_obs}  seed={args.seed}\n")
                quantized_log_fh.write(f"# use_fixed_pi0_noise={bool(use_fixed_pi0_noise)}\n")
                quantized_log_fh.write("# quantized_server_cmd=None (evaluate existing server)\n\n")
                if base_q:
                    quantized_log_fh.write(f"# base_quant={base_q}\n\n")
                quantized_log_fh.flush()

            quantized = _ws.WebsocketClientPolicy(host=args.quantized_host, port=args.quantized_port)
            _wait_until_ready(quantized, obs0, timeout_s=args.ready_timeout_s)

            base_actions: list[torch.Tensor] = []
            quantized_actions: list[torch.Tensor] = []
            for obs in observations:
                base_actions.append(_to_actions_tensor(base.infer(obs)))
                quantized_actions.append(_to_actions_tensor(quantized.infer(obs)))

            m = _metrics(base_actions, quantized_actions)
            tag = f"rel_err={rel_err:.4e}" if rel_err is not None else "rel_err=<as-is>"
            line = f"{tag:20s}  rmse={m.rmse:.4e}"
            print(line)
            run_log.write(line + "\n")
            run_log.flush()
            if quantized_log_fh is not None:
                quantized_log_fh.write(f"\n# result: {line}\n")
                quantized_log_fh.flush()

            if rel_err is not None and m.rmse >= args.rmse_threshold:
                print(f"STOP: threshold violated (rmse >= {args.rmse_threshold})")
                break

            if math.isnan(m.rmse):
                consecutive_nan += 1
                if consecutive_nan >= 3:
                    print("STOP: 3 consecutive NaN RMSE — moving to next combo")
                    break
            else:
                consecutive_nan = 0

            if args.quantized_server_cmd is None:
                break
    finally:
        run_log.close()
        if quantized_proc is not None and quantized_proc.poll() is None:
            _stop_proc_tree(quantized_proc)
        if quantized_log_fh is not None:
            quantized_log_fh.close()


if __name__ == "__main__":
    main()
