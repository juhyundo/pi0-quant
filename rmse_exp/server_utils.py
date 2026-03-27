"""
server_utils.py
---------------
Shared utilities for rmse_exp sweep scripts.

Port / process management:
    _wait_for_port
    _pids_listening_on_port
    _kill_listeners_on_port
    _stop_proc_tree
    _wait_until_ready

Observation helpers:
    _random_observation_droid
    _with_fixed_pi0_noise
    _to_actions_tensor

Log helpers:
    _timestamp_tag
    _open_step_log
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import IO, Any, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Port / process management
# ---------------------------------------------------------------------------

def _wait_for_port(port: int, *, timeout_s: float = 120.0, interval_s: float = 1.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(interval_s)
    return False


def _pids_listening_on_port(port: int) -> set[int]:
    def _listen_inodes_from(path: str) -> set[str]:
        inodes: set[str] = set()
        with open(path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.split()
                if len(parts) < 10:
                    continue
                local_address = parts[1]
                st = parts[3]
                inode = parts[9]
                if st != "0A":
                    continue
                try:
                    _ip_hex, port_hex = local_address.split(":")
                    p = int(port_hex, 16)
                except Exception:
                    continue
                if p == port:
                    inodes.add(inode)
        return inodes

    inodes: set[str] = set()
    with suppress(FileNotFoundError, PermissionError):
        inodes |= _listen_inodes_from("/proc/net/tcp")
    with suppress(FileNotFoundError, PermissionError):
        inodes |= _listen_inodes_from("/proc/net/tcp6")
    if not inodes:
        return set()

    pids: set[int] = set()
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        fd_dir = f"/proc/{pid_str}/fd"
        try:
            fds = os.listdir(fd_dir)
        except (FileNotFoundError, PermissionError):
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inode = target[len("socket:["):-1]
                if inode in inodes:
                    pids.add(pid)
                    break
    return pids


def _kill_listeners_on_port(port: int, *, timeout_s: float = 10.0) -> None:
    pids = _pids_listening_on_port(port)
    if not pids:
        return
    for pid in pids:
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not _pids_listening_on_port(port):
            return
        time.sleep(0.1)
    for pid in _pids_listening_on_port(port):
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def _stop_proc_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=15)


def _wait_until_ready(policy: Any, obs: dict, *, timeout_s: float) -> None:
    t0 = time.time()
    last_err: Optional[BaseException] = None
    while True:
        try:
            policy.infer(obs)
            return
        except BaseException as e:
            last_err = e
            if time.time() - t0 >= timeout_s:
                raise RuntimeError(f"Server not ready after {timeout_s:.1f}s") from last_err
            time.sleep(0.25)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _random_observation_droid(rng: np.random.Generator) -> dict:
    return {
        "observation/exterior_image_1_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left":      rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position":        rng.random(7, dtype=np.float32),
        "observation/gripper_position":      rng.random(1, dtype=np.float32),
        "prompt": "Grab the object",
    }


def _with_fixed_pi0_noise(
    obs: dict,
    *,
    rng: np.random.Generator,
    action_horizon: int,
    action_dim: int,
) -> dict:
    out = dict(obs)
    out["pi0_noise"] = rng.standard_normal((action_horizon, action_dim)).astype(np.float32)
    return out


def _to_actions_tensor(resp: dict) -> torch.Tensor:
    a = resp["actions"]
    return torch.from_numpy(np.asarray(a).copy()).float().reshape(-1)


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _timestamp_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _open_step_log(*, log_dir: Path, tag: str) -> IO[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    return (log_dir / f"{tag}.log").open("w", encoding="utf-8")
