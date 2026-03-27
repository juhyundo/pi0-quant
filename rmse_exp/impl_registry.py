"""
impl_registry.py
----------------
Maps implementation aliases to their serve_quant.py paths and capability flags.

Aliases
-------
  base       pi0_inout            (no rel_err support)
  ipt_c      pi0_inout_c          (no rel_err support; C-RTL backend, E4M3-only)
  ipt_mxu    pi0_inout_ipt_mxu    (rel_err supported)
  ipt_numba  pi0_inout_ipt_mxu_numba  (rel_err supported, fp8_mode=mx default)
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

IMPLS: dict[str, dict] = {
    "base": {
        "serve_quant":      _REPO_ROOT / "pi0_inout" / "serve_quant.py",
        "supports_rel_err": False,
        "default_fp8_mode": "mx",
    },
    "ipt_c": {
        "serve_quant":      _REPO_ROOT / "pi0_inout_c" / "serve_quant.py",
        "supports_rel_err": True,
        "default_fp8_mode": "mx",
    },
    "ipt_mxu": {
        "serve_quant":      _REPO_ROOT / "pi0_inout_ipt_mxu" / "serve_quant.py",
        "supports_rel_err": True,
        "default_fp8_mode": "mx",
    },
    "ipt_numba": {
        "serve_quant":      _REPO_ROOT / "pi0_inout_ipt_mxu_numba" / "serve_quant.py",
        "supports_rel_err": True,
        "default_fp8_mode": "mx",
    },
}


def get_impl(alias: str) -> dict:
    if alias not in IMPLS:
        raise ValueError(f"Unknown impl {alias!r}. Valid: {list(IMPLS)}")
    return IMPLS[alias]


def require_rel_err(alias: str) -> dict:
    """Return impl dict or raise if rel_err is not supported."""
    impl = get_impl(alias)
    if not impl["supports_rel_err"]:
        raise ValueError(
            f"Impl {alias!r} does not support --rel-err noise injection. "
            f"Use one of: {[k for k, v in IMPLS.items() if v['supports_rel_err']]}"
        )
    return impl
