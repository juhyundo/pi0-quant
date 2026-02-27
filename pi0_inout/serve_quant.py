"""
serve_quant.py
--------------
Drop-in replacement for openpi/scripts/serve_policy.py that serves
PI0Pytorch with configurable matmul input/output quantization.

How it works
------------
1. Adds openpi/src and openpi-client/src to sys.path.
2. Injects lightweight Python stubs for the three openpi files that import
   JAX at module level (gemma config, lora config, array_typing, image_tools).
   This lets PI0Pytorch be imported in the pi0 conda env (torch-only).
3. Instantiates PI0Pytorch from a hardcoded config dict for the known
   training configs (no JAX config system required).
4. Loads weights from a local safetensors checkpoint if available.
5. Patches every nn.Linear with QuantLinear (input_fmt / output_fmt).
6. Serves via the openpi WebSocket protocol (msgpack + websockets).
7. On SIGTERM/SIGINT, writes per-layer RMSE stats to --stats-output.

Quantization semantics (unchanged from quant_linear.py)
---------------------------------------------------------
For each nn.Linear:
    x_q  = quant(x,    input_fmt)
    W_q  = quant(W,    input_fmt)
    b_q  = quant(bias, input_fmt)          # bias loaded in input_fmt
    y    = F.linear(x_q, W_q, b_q)        # float32 accumulation
    out  = quant(y, output_fmt)            # single output quantization

FLOAT32/FLOAT32 is the identity — zero RMSE baseline.

Usage
-----
    python serve_quant.py \\
        --openpi-dir /path/to/openpi \\
        --checkpoint-dir /path/to/model.safetensors_dir \\
        --config pi05_droid_jointpos_polaris \\
        --input-fmt float8_e4m3 --output-fmt float16 \\
        --port 8003 --gpu 0
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import atexit
import json
import logging
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Make sure openpi source and openpi-client are importable ─────────────────
_THIS_DIR    = Path(__file__).resolve().parent
_OPENPI_DIR  = Path(os.environ.get("OPENPI_DIR", _THIS_DIR.parent / "openpi"))
_OPENPI_SRC  = _OPENPI_DIR / "src"
_CLIENT_SRC  = _OPENPI_DIR / "packages" / "openpi-client" / "src"
_PI0_INOUT   = _THIS_DIR          # for "from pi0_inout import ..."

for _p in [str(_PI0_INOUT.parent), str(_CLIENT_SRC), str(_OPENPI_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Inject JAX stubs BEFORE any openpi import ────────────────────────────────
from pi0_inout._jax_stubs import inject as _inject_jax_stubs   # noqa: E402
_inject_jax_stubs()

# ── Now it is safe to import the pytorch model ────────────────────────────────
import torch
import torch.nn as nn

# ── pi0_inout imports (quantization layer) ────────────────────────────────────
from pi0_inout.quant_types import QuantFormat
from pi0_inout.model_patcher import patch_model, patch_model_matvec, list_linear_layers
from pi0_inout.stats_tracker import StatsTracker
from pi0_inout.ulp_noise import UlpNoiseConfig


# ---------------------------------------------------------------------------
# Known training configs  (avoids importing openpi.training.config / JAX)
# ---------------------------------------------------------------------------

# Each entry: SimpleNamespace with the fields PI0Pytorch.__init__ reads.
# Add new configs here as needed.
_KNOWN_CONFIGS: dict[str, SimpleNamespace] = {
    # pi05 droid joint-position policy (Polaris)
    "pi05_droid_jointpos_polaris": SimpleNamespace(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        dtype="bfloat16",
        action_dim=32,
        action_horizon=15,
        max_token_len=200,
    ),
    # pi05 droid (non-Polaris)
    "pi05_droid": SimpleNamespace(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        dtype="bfloat16",
        action_dim=32,
        action_horizon=15,
        max_token_len=200,
    ),
    # pi0 droid joint-position (non-pi05)
    "pi0_droid": SimpleNamespace(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
        dtype="bfloat16",
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
    ),
    # aloha sim (non-pi05)
    "pi0_aloha_sim": SimpleNamespace(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
        dtype="bfloat16",
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
    ),
}


def _get_model_config(config_name: str) -> SimpleNamespace:
    if config_name in _KNOWN_CONFIGS:
        return _KNOWN_CONFIGS[config_name]
    # Fallback: pi05 defaults
    logger.warning(
        f"Config '{config_name}' not in _KNOWN_CONFIGS; using pi05 defaults. "
        f"Known: {list(_KNOWN_CONFIGS)}"
    )
    return _KNOWN_CONFIGS["pi05_droid_jointpos_polaris"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pi0_pytorch(
    config_name: str,
    checkpoint_dir: str,
    device: torch.device,
) -> nn.Module:
    """
    Load PI0Pytorch without JAX.

    1. Constructs the model config from _KNOWN_CONFIGS.
    2. Instantiates PI0Pytorch (stubs handle JAX imports).
    3. Loads weights from a local safetensors file if available.
       Falls back to random init with a warning if no checkpoint found.
    """
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    cfg = _get_model_config(config_name)
    logger.info(
        f"Instantiating PI0Pytorch: pi05={cfg.pi05}, "
        f"paligemma={cfg.paligemma_variant}, expert={cfg.action_expert_variant}, "
        f"dtype={cfg.dtype}, action_horizon={cfg.action_horizon}"
    )

    model = PI0Pytorch(cfg)
    # torch.compile is applied in __init__; QuantLinear has Python-level conditional
    # logic that Dynamo cannot trace. Unwrap back to the raw Python method.
    model.sample_actions = model.sample_actions.__wrapped__
    model = model.to(device)
    model.eval()

    _load_checkpoint(model, checkpoint_dir, device)
    return model


def _load_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    device: torch.device,
) -> None:
    """
    Load weights into the model.  Tries (in order):
      1. <checkpoint_dir>/model.safetensors   (train_pytorch.py output)
      2. <checkpoint_dir>/pytorch_model.pt
      3. <checkpoint_dir>/pytorch_model.bin
    Falls back to a warning (random init) if none found.
    GCS paths (gs://) are skipped — download them first with gsutil.
    """
    if checkpoint_dir.startswith("gs://"):
        logger.warning(
            f"Checkpoint is a GCS path: {checkpoint_dir}\n"
            "  GCS checkpoints are JAX/Orbax format and require JAX to load.\n"
            "  Convert to safetensors first with:\n"
            "    python /path/to/openpi/examples/convert_jax_model_to_pytorch.py \\\n"
            f"        --checkpoint_dir <local_download_of_{checkpoint_dir}> \\\n"
            f"        --config_name pi05_droid_jointpos_polaris \\\n"
            "        --output_path /path/to/checkpoints/pi05_droid_pytorch\n"
            "  Then pass --checkpoint-dir /path/to/checkpoints/pi05_droid_pytorch\n"
            "  Running with RANDOM WEIGHTS (RMSE results will still be meaningful\n"
            "  for quantization analysis relative to the fp32 baseline)."
        )
        return

    ckpt_dir = Path(checkpoint_dir)

    # safetensors (preferred)
    sf_path = ckpt_dir / "model.safetensors"
    if sf_path.exists():
        try:
            import safetensors.torch
            safetensors.torch.load_model(model, str(sf_path), device=str(device))
            logger.info(f"Loaded weights from {sf_path}")
            return
        except Exception as e:
            logger.warning(f"safetensors load failed: {e}")

    # legacy PyTorch save
    for candidate in ["pytorch_model.pt", "pytorch_model.bin", "model.pt"]:
        pt_path = ckpt_dir / candidate
        if pt_path.exists():
            state = torch.load(str(pt_path), map_location=device)
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded weights from {pt_path}")
            return

    logger.warning(
        f"No checkpoint found in {checkpoint_dir}. "
        "Running with RANDOM WEIGHTS — RMSE will still correctly measure "
        "quantization error relative to the fp32 baseline, but the benchmark "
        "success rates and videos will not reflect the real model's behaviour."
    )


# ---------------------------------------------------------------------------
# Policy shim (Pi0PyTorchPolicy)
# ---------------------------------------------------------------------------

class Pi0PyTorchPolicy:
    """
    Adapts PI0Pytorch to the openpi policy interface expected by
    WebsocketPolicyServer:
        policy.infer(obs_dict)  → {"actions": np.ndarray}
        policy.metadata         → dict
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model    = model
        self.device   = device
        self.metadata = {"model": "PI0Pytorch", "quantized": True}

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map DROID WebSocket obs dict → PI0Pytorch SimpleNamespace and run inference.

        Client sends:
          "observation/exterior_image_1_left"  uint8 HWC [224,224,3]  → base_0_rgb
          "observation/wrist_image_left"        uint8 HWC [224,224,3]  → left_wrist_0_rgb
          "observation/joint_position"          float [6]
          "observation/gripper_position"        float [1]
          "prompt"                              str (ignored — zero tokens)

        right_wrist_0_rgb is not sent by the client; we fill with zeros and mask=False.
        """
        import numpy as np

        dev = self.device
        dtype = torch.float32

        def _img_tensor(arr: np.ndarray) -> torch.Tensor:
            """uint8 HWC [H,W,3] → float tensor [1,3,H,W] (preprocessing expects CHW)."""
            t = torch.from_numpy(arr.copy()).to(dev)          # [H,W,3] uint8
            t = t.permute(2, 0, 1).unsqueeze(0).to(dtype)    # [1,3,H,W]
            return t

        H, W = 224, 224

        base_img  = _img_tensor(obs["observation/exterior_image_1_left"])
        wrist_img = _img_tensor(obs["observation/wrist_image_left"])
        zero_img  = torch.zeros(1, 3, H, W, dtype=dtype, device=dev)

        images = {
            "base_0_rgb":        base_img,
            "left_wrist_0_rgb":  wrist_img,
            "right_wrist_0_rgb": zero_img,
        }
        image_masks = {
            "base_0_rgb":        torch.ones(1,  dtype=torch.bool, device=dev),
            "left_wrist_0_rgb":  torch.ones(1,  dtype=torch.bool, device=dev),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=dev),
        }

        # State: 7-dim robot state (6 joints + 1 gripper) padded to action_dim=32
        joint = torch.from_numpy(np.array(obs["observation/joint_position"], dtype=np.float32)).to(dev)
        grip  = torch.from_numpy(np.array(obs["observation/gripper_position"], dtype=np.float32)).to(dev)
        raw_state = torch.cat([joint.flatten(), grip.flatten()])   # [7]
        state = torch.zeros(1, 32, dtype=dtype, device=dev)
        state[0, :raw_state.shape[0]] = raw_state

        # Tokenised prompt: zeros (mask=0 → model ignores language tokens)
        max_tok = 200
        tokenized_prompt      = torch.zeros(1, max_tok, dtype=torch.int64,  device=dev)
        tokenized_prompt_mask = torch.zeros(1, max_tok, dtype=torch.bool,   device=dev)

        obs_ns = SimpleNamespace(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            # openpi preprocessing expects these attributes to exist 
            token_ar_mask=None,
            token_loss_mask=None,
        )

        # Deterministic noise
        noise = None
        if "pi0_noise" in obs and obs["pi0_noise"] is not None:
            n = np.asarray(obs["pi0_noise"], dtype=np.float32)
            if n.ndim == 2:
                n = n[None, ...]
            noise = torch.from_numpy(n.copy()).to(dev)

        with torch.no_grad():
            actions = self.model.sample_actions(str(dev), obs_ns, noise=noise, num_steps=10)
        # actions: [1, action_horizon=15, action_dim=32]
        return {"actions": actions.squeeze(0).cpu().numpy()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        force=True,
    )
    args = parse_args()

    # Propagate openpi dir to stubs (in case it was set via CLI)
    if args.openpi_dir:
        global _OPENPI_DIR, _OPENPI_SRC, _CLIENT_SRC
        _OPENPI_DIR = Path(args.openpi_dir)
        _OPENPI_SRC = _OPENPI_DIR / "src"
        _CLIENT_SRC = _OPENPI_DIR / "packages" / "openpi-client" / "src"
        for _p in [str(_CLIENT_SRC), str(_OPENPI_SRC)]:
            if _p not in sys.path:
                sys.path.insert(0, _p)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Load and optionally list layers ──────────────────────────────────
    logger.info(f"Loading model: config={args.config}  ckpt={args.checkpoint_dir}")
    model = load_pi0_pytorch(args.config, args.checkpoint_dir, device)

    if args.list_layers:
        rows = list_linear_layers(model)
        print(f"\n{'Component':16s}  {'In':6s}  {'Out':6s}  Name")
        print("-" * 80)
        for r in rows:
            print(f"{r['component']:16s}  {r['in_features']:6d}  {r['out_features']:6d}  {r['name']}")
        print(f"\nTotal linear layers: {len(rows)}")
        return

    # ── Parse formats and patch model ────────────────────────────────────
    input_fmt  = QuantFormat(args.input_fmt)
    output_fmt = QuantFormat(args.output_fmt)

    # ULP noise injection into Linear matmul outputs
    ulp_noise = None
    if args.ulp_n and args.ulp_n > 0:
        ulp_noise = UlpNoiseConfig(
            n_ulp=args.ulp_n,
            ulp_fmt=QuantFormat(args.ulp_fmt),
        )

    tracker = StatsTracker()
    if args.use_matvec:
        # Matrix/vector mode: enforce vec_in == mat_out internally.
        patch_model_matvec(
            model=model,
            matrix_in_fmt=QuantFormat(args.mat_in_fmt),
            matrix_out_fmt=QuantFormat(args.mat_out_fmt),
            vector_out_fmt=QuantFormat(args.vec_out_fmt),
            tracker=tracker,
            ulp_noise=ulp_noise,
            verbose=False,
        )
        logger.info(
            "Model patched (matvec): "
            f"mat_in={args.mat_in_fmt} mat_out={args.mat_out_fmt} vec_out={args.vec_out_fmt} "
            f"ulp_n={args.ulp_n} ulp_fmt={args.ulp_fmt}"
        )
    else:
        patch_model(
            model=model,
            input_fmt=input_fmt,
            output_fmt=output_fmt,
            tracker=tracker,
            ulp_noise=ulp_noise,
            verbose=False,
        )
        logger.info(
            f"Model patched: input_fmt={input_fmt.value}  output_fmt={output_fmt.value}  "
            f"ulp_n={args.ulp_n} ulp_fmt={args.ulp_fmt}"
        )

    # ── Register stats dump on exit ───────────────────────────────────────
    def _dump_stats() -> None:
        logger.info("=== Quantization RMSE Report ===")
        report = tracker.summary()
        report.print(show_layers=False)
        if args.stats_output:
            Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.stats_output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Stats saved to {args.stats_output}")

    atexit.register(_dump_stats)
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # ── Build policy and start WebSocket server ───────────────────────────
    policy = Pi0PyTorchPolicy(model=model, device=device)
    policy.metadata = {
        "model": "PI0Pytorch",
        "quantized": True,
        "quant": {
            "input_fmt": input_fmt.value,
            "output_fmt": output_fmt.value,
            "use_matvec": bool(args.use_matvec),
            "mat_in_fmt": getattr(args, "mat_in_fmt", None),
            "mat_out_fmt": getattr(args, "mat_out_fmt", None),
            "vec_out_fmt": getattr(args, "vec_out_fmt", None),
            "ulp_n": int(getattr(args, "ulp_n", 0) or 0),
            "ulp_fmt": getattr(args, "ulp_fmt", None),
        },
    }

    from openpi.serving import websocket_policy_server
    import socket
    logger.info(f"Starting server on {socket.gethostname()}:{args.port}  "
                f"(input={input_fmt.value}, output={output_fmt.value})")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve PI0Pytorch with matmul quantization over WebSocket."
    )
    p.add_argument("--openpi-dir", default=None,
                   help="Path to the openpi repository root (default: ../openpi relative to this file)")
    p.add_argument("--config", default="pi05_droid_jointpos_polaris",
                   help="Training config name (used to look up architecture params)")
    p.add_argument("--checkpoint-dir", default="",
                   help="Directory containing model.safetensors (or gs:// path with a warning)")
    p.add_argument("--port",  type=int, default=8003)
    p.add_argument("--gpu",   type=int, default=0,
                   help="CUDA device index (-1 for CPU)")

    # Quantization
    p.add_argument("--input-fmt",  default="float32",
                   choices=[f.value for f in QuantFormat])
    p.add_argument("--output-fmt", default="float32",
                   choices=[f.value for f in QuantFormat])

    # Optional: matrix/vector separate formats (constraint: vec_in == mat_out)
    p.add_argument("--use-matvec", action="store_true",
                   help="Use matrix/vector IO formats for nn.Linear instead of simple input/output formats")
    p.add_argument("--mat-in-fmt", default="float32",
                   choices=[f.value for f in QuantFormat],
                   help="Matrix input format (activation+weight)")
    p.add_argument("--mat-out-fmt", default="float32",
                   choices=[f.value for f in QuantFormat],
                   help="Matrix output format (matmul output before bias add)")
    p.add_argument("--vec-out-fmt", default="float32",
                   choices=[f.value for f in QuantFormat],
                   help="Vector output format (final output after bias add)")

    # Optional: ULP noise injection into matmul outputs
    p.add_argument("--ulp-n", type=int, default=0,
                   help="Inject +/- n ULP noise into each Linear matmul output (0 disables)")
    p.add_argument("--ulp-fmt", default="bfloat16",
                   choices=[f.value for f in QuantFormat],
                   help="Format whose ULP grid defines the step size")

    # Output
    p.add_argument("--stats-output", default=None,
                   help="Write JSON RMSE stats here on exit")
    p.add_argument("--list-layers", action="store_true",
                   help="Print linear layer inventory and exit")
    return p.parse_args()


if __name__ == "__main__":
    main()
