"""
analyze_timing.py
-----------------
Parse __call__: done lines from a CIPTLinearRTLFunction server log and produce:
  1. timing_bars.png   — avg/min/max elapsed per (component, w_shape) group
  2. timing_timeline.png — elapsed per call in sequence, colored by component

Usage:
    uv run python rmse_exp/analyze_timing.py <path/to/server.log>
    uv run python rmse_exp/analyze_timing.py <path/to/server.log> --out-dir /tmp/plots
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Format produced by ipt_rtl_linear_c.py:
#   __call__: done  [vision] enc.0.mlp.fc1  elapsed=0.923s  w(1152,1152)  batch=256  fmt=OutBF16
_DONE_RE = re.compile(
    r"__call__: done\s+"
    r"\[(?P<component>[^\]]+)\]\s+(?P<layer>[^\s]+)\s+"
    r"elapsed=(?P<elapsed>[0-9.]+)s\s+"
    r"w\((?P<out>\d+),(?P<in_>\d+)\)\s+"
    r"batch=(?P<batch>\d+)"
)


def parse_log(log_path: Path) -> list[dict]:
    records = []
    with open(log_path) as f:
        for line in f:
            m = _DONE_RE.search(line)
            if not m:
                continue
            records.append({
                "component": m.group("component"),
                "layer":     m.group("layer"),
                "elapsed":   float(m.group("elapsed")),
                "w_shape":   (int(m.group("out")), int(m.group("in_"))),
                "batch":     int(m.group("batch")),
                "call_idx":  len(records),
            })
    return records


# ---------------------------------------------------------------------------
# Bar chart: avg/min/max per (component, w_shape)
# ---------------------------------------------------------------------------

COMPONENT_COLORS = {
    "vision":      "#4C72B0",
    "language":    "#DD8452",
    "transformer": "#DD8452",
    "action_expert": "#55A868",
    "action_head": "#C44E52",
}

def _color(component: str) -> str:
    for k, v in COMPONENT_COLORS.items():
        if k in component.lower():
            return v
    return "#8172B2"


def plot_bars(records: list[dict], out_path: Path) -> None:
    # Group by (component, w_shape)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        key = (r["component"], r["w_shape"])
        groups[key].append(r["elapsed"])

    # Sort by mean elapsed descending
    sorted_groups = sorted(groups.items(), key=lambda kv: np.mean(kv[1]), reverse=True)

    labels = [f"[{comp}]\n{shape[0]}×{shape[1]}" for (comp, shape), _ in sorted_groups]
    means  = [np.mean(v)  for _, v in sorted_groups]
    mins_  = [np.min(v)   for _, v in sorted_groups]
    maxs_  = [np.max(v)   for _, v in sorted_groups]
    counts = [len(v)      for _, v in sorted_groups]
    colors = [_color(comp) for (comp, _), _ in sorted_groups]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.7), 6))

    bars = ax.bar(x, means, color=colors, alpha=0.85, zorder=3, label="mean")
    ax.errorbar(x, means,
                yerr=[np.array(means) - np.array(mins_),
                      np.array(maxs_) - np.array(means)],
                fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)

    # Annotate with call count
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"n={n}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Elapsed (s)")
    ax.set_title("C RTL layer timing: avg / min / max per (component, weight shape)")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Legend for components
    seen = {}
    for (comp, _), _ in sorted_groups:
        if comp not in seen:
            seen[comp] = _color(comp)
    patches = [mpatches.Patch(color=c, label=comp) for comp, c in seen.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Timeline plot
# ---------------------------------------------------------------------------

def plot_timeline(records: list[dict], out_path: Path) -> None:
    if not records:
        return

    call_idxs = [r["call_idx"] for r in records]
    elapseds  = [r["elapsed"]  for r in records]
    components = [r["component"] for r in records]

    # Assign a color per component
    unique_comps = list(dict.fromkeys(components))
    cmap = plt.colormaps.get_cmap("tab10")
    comp_color = {c: cmap(i / max(len(unique_comps), 1)) for i, c in enumerate(unique_comps)}
    colors = [comp_color[c] for c in components]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(call_idxs, elapseds, c=colors, s=18, alpha=0.75, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Call index (sequential order across full inference)")
    ax.set_ylabel("Elapsed (s)  [log scale]")
    ax.set_title("C RTL call timeline — elapsed per layer call")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    patches = [mpatches.Patch(color=comp_color[c], label=c) for c in unique_comps]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(records: list[dict]) -> None:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        groups[(r["component"], r["w_shape"])].append(r["elapsed"])

    sorted_groups = sorted(groups.items(), key=lambda kv: np.mean(kv[1]), reverse=True)

    print(f"\n{'component':<16} {'w_shape':<14} {'n':>5} {'mean':>8} {'min':>8} {'max':>8}  total")
    print("-" * 72)
    for (comp, shape), vals in sorted_groups:
        print(f"{comp:<16} {str(shape):<14} {len(vals):>5} "
              f"{np.mean(vals):>8.2f}s {np.min(vals):>8.2f}s {np.max(vals):>8.2f}s "
              f"{sum(vals):>8.1f}s")
    total = sum(r["elapsed"] for r in records)
    print(f"\nTotal measured: {total:.1f}s across {len(records)} calls")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Analyze CIPTLinearRTLFunction timing logs")
    p.add_argument("log", type=Path, help="Path to server log file")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for plots (default: same dir as log)")
    args = p.parse_args()

    if not args.log.exists():
        print(f"Error: {args.log} not found", file=sys.stderr)
        sys.exit(1)

    records = parse_log(args.log)
    if not records:
        print("No '__call__: done' lines found. Make sure the log includes component/layer/elapsed fields.")
        sys.exit(1)

    print(f"Parsed {len(records)} calls from {args.log}")
    print_summary(records)

    out_dir = args.out_dir or args.log.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.log.stem
    plot_bars(records,     out_dir / f"{stem}_timing_bars.png")
    plot_timeline(records, out_dir / f"{stem}_timing_timeline.png")


if __name__ == "__main__":
    main()
