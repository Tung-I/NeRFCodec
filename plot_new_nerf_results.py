"""
Usage:
python plot_new_nerf_results.py \
    --log-scale \
    --out nerf_rd_curve.png \
    --empty-out nerf_rd_curve_empty.png \
    --no-show
"""

#!/usr/bin/env python3
import argparse
import itertools
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Global experiment configuration (merged from nerf_results.json)
# Edit here directly instead of using a JSON file.
# ============================================================
NERF_SYNTHETIC_CFG = {
    "experiment_name": "NeRF-Synthetic Dataset",
    "x_label": "Size in log (MB)",  # log10-scale axis
    "y_label": "PSNR (dB)",

    # ---- Figure / axes geometry ----
    "fig_width": 10.0,       # inches: full figure width
    "fig_height": 4.0,      # inches: full figure height
    # Fraction of figure width occupied by the axes.
    # Smaller -> ticks (10^-0.5 ... 10^3) closer together.
    "axes_width_frac": 0.48,   # <=== TWEAK THIS to change spacing between 10^k
    "axes_height_frac": 0.75, # can tweak vertical extent if you want

    # ---- Custom log10-x axis design ----
    # We plot x_plot = log10(bitrate).
    "x_exp_min": -0.5,                    # corresponds to 10^{-0.5}
    "x_exp_max": 3.0,                     # corresponds to 10^{3}
    "x_ticks_exp": [-0.5, 0.0, 1.0, 2.0, 3.0],
    "x_grid_exp": [0.0, 1.0, 2.0],        # vertical grid lines at 10^0, 10^1, 10^2

    # ---- Custom y axis design ----
    "y_min": 30.9,
    "y_max": 33.5,                        # your requested upper bound
    "y_ticks": [31.0, 32.0, 33.0, 33.5],
    "y_grid": [32.0, 33.0],               # horizontal grid lines at 32 and 33

    # ---- Line and border style (grid + boundaries) ----
    "border_color": "0.8",     # boundary (axes spines) color
    "border_width": 0.8,
    "grid_color": "0.8",       # grid line color
    "grid_width": 0.8,
    "grid_linestyle": "-",     # solid gray grid, same as boundaries

    "metric_key": "metric",
    "show_markers": True,

    # ---- Methods and points (your data) ----
    "methods": [
        {
            "name": "CATRF-JPEG",
            "linestyle": "-",
            "marker": "o",
            "color": "red",
            "points": [
                { "plane_rate_mb": 0.764, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 32.90 },
                { "plane_rate_mb": 0.600, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 32.68 },
                { "plane_rate_mb": 0.479, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 32.49 },
                { "plane_rate_mb": 0.357, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 32.17 }
            ]
        },
        {
            "name": "CNC",
            "linestyle": "-",
            "marker": "o",
            "color": "dodgerblue",
            "points": [
                { "model_rate_mb": 0.991, "metric": 33.16 },
                { "model_rate_mb": 0.850, "metric": 32.68 },
                { "model_rate_mb": 0.765, "metric": 32.49 },
                { "model_rate_mb": 0.613, "metric": 31.70 }
            ]
        },
        {
            "name": "NeRFCodec",
            "linestyle": "-",
            "marker": "o",
            "color": "orange",
            "points": [
                { "model_rate_mb": 1.842, "metric": 32.19 },
                { "codec_rate_mb": 1.0879, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 31.68 }
            ]
        },
        {
            "name": "BiRF",
            "linestyle": "-",
            "marker": "o",
            "color": "brown",
            "points": [
                { "model_rate_mb": 2.8, "metric": 33.26 },
                { "model_rate_mb": 1.4, "metric": 32.64 },
                { "model_rate_mb": 0.7, "metric": 31.53 }
            ]
        },
        {
            "name": "PPNG",
            "linestyle": "-",
            "marker": "o",
            "color": "gray",
            "points": [
                { "model_rate_mb": 32.8, "metric": 31.90 },
                { "model_rate_mb": 2.49, "metric": 30.99 },
                # { "model_rate_mb": 0.151, "metric": 28.89 }
            ]
        },
        {
            "name": "ECRF",
            "linestyle": "-",
            "marker": "o",
            "color": "green",
            "points": [
                { "model_rate_mb": 2.5, "metric": 33.05 },
                { "model_rate_mb": 1.3, "metric": 32.72 },
                { "model_rate_mb": 0.8, "metric": 32.20 }
            ]
        },
        {
            "name": "Masked-Wavelet",
            "linestyle": "-",
            "marker": "o",
            "color": "purple",
            "points": [
                { "model_rate_mb": 2.4, "metric": 32.38 },
                { "model_rate_mb": 1.5, "metric": 32.23 },
                { "model_rate_mb": 0.8, "metric": 31.95 }
            ]
        },
        {
            "name": "TensoRF-VM",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 67.5, "metric": 33.05 }
            ]
        },
        {
            "name": "Instant-NGP",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 39.5, "metric": 33.08 }
            ]
        },
        {
            "name": "TensoRF-CP",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 3.9, "metric": 31.66 }
            ]
        },
        {
            "name": "TensoRF High",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 7.9, "metric": 32.81 }
            ]
        },
        # {
        #     "name": "VQ-DVGO",
        #     "linestyle": "None",
        #     "marker": "o",
        #     "points": [
        #         { "model_rate_mb": 1.4, "metric": 31.77 }
        #     ]
        # },
        {
            "name": "VQ-TensoRF",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 3.5, "metric": 32.88 }
            ]
        },
        {
            "name": "K-Planes-hybrid",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 300, "metric": 32.3 }
            ]
        },
        {
            "name": "DVGO",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 600, "metric": 31.95 }
            ]
        },
        {
            "name": "Plenoxels",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 800, "metric": 31.8 }
            ]
        }
    ]
}

# Pick which config to use
CFG = NERF_SYNTHETIC_CFG

# ============================================================
# Helpers
# ============================================================
def sum_rate_mb(point: Dict) -> float:
    """Sum all keys ending with '_rate_mb'."""
    total = 0.0
    for k, v in point.items():
        if isinstance(v, (int, float)) and k.endswith("_rate_mb"):
            total += float(v)
    return total

def extract_xy(points: List[Dict], metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rates, metrics) sorted by rate."""
    xy = []
    for p in points:
        if metric_key not in p:
            raise KeyError(f"Point missing '{metric_key}': {p}")
        rate = sum_rate_mb(p)
        xy.append((rate, float(p[metric_key])))
    xy.sort(key=lambda t: t[0])
    rates = np.array([r for r, _ in xy], dtype=float)
    metrics = np.array([m for _, m in xy], dtype=float)
    return rates, metrics

def auto_style(i: int):
    """Fallback color/marker/linestyle if not specified per method."""
    colors = itertools.cycle(
        plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])
    )
    markers = itertools.cycle(['o','s','^','D','v','P','X','*','<','>'])
    linestyles = itertools.cycle(['-','--','-.',':'])
    c = None; m = None; ls = None
    for _ in range(i + 1):
        c = next(colors); m = next(markers); ls = next(linestyles)
    return c, m, ls

def format_log_label(exp: float) -> str:
    """Format tick label as 10^{exp}, allowing half steps like -0.5."""
    if abs(exp - int(round(exp))) < 1e-6:
        e_int = int(round(exp))
        return rf"$10^{e_int}$"
    else:
        return rf"$10^{{{exp:.1f}}}$"

def create_figure_and_axes(cfg: Dict):
    """Create figure and manually-sized axes (so we can control width)."""
    fig_w = cfg.get("fig_width", 7.0)
    fig_h = cfg.get("fig_height", 4.8)
    axes_w_frac = cfg.get("axes_width_frac", 0.7)
    axes_h_frac = cfg.get("axes_height_frac", 0.75)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Center the axes box horizontally/vertically
    left = (1.0 - axes_w_frac) / 2.0
    bottom = (1.0 - axes_h_frac) / 2.0
    ax = fig.add_axes([left, bottom, axes_w_frac, axes_h_frac])

    return fig, ax

def setup_axes(ax: plt.Axes, cfg: Dict, use_log_axis: bool):
    """Configure axes limits, ticks, and grid per requirements."""
    x_label = cfg.get("x_label", "Bitrate (MB)")
    y_label = cfg.get("y_label", "Metric")

    border_color = cfg.get("border_color", "0.0")
    border_width = cfg.get("border_width", 1.0)
    grid_color   = cfg.get("grid_color", border_color)
    grid_width   = cfg.get("grid_width", border_width)
    grid_ls      = cfg.get("grid_linestyle", "-")

    # ---- X axis setup ----
    if use_log_axis:
        x_exp_min = cfg.get("x_exp_min", -0.5)
        x_exp_max = cfg.get("x_exp_max", 3.0)
        x_ticks_exp = cfg.get("x_ticks_exp", [-0.5, 0.0, 1.0, 2.0, 3.0])

        ax.set_xlim(x_exp_min, x_exp_max)
        ax.set_xticks(x_ticks_exp)
        ax.set_xticklabels([format_log_label(e) for e in x_ticks_exp], fontsize=12)
    else:
        x_limits = cfg.get("x_limits", None)
        if x_limits:
            ax.set_xlim(x_limits[0], x_limits[1])

    # ---- Y axis setup ----
    y_min = cfg.get("y_min", None)
    y_max = cfg.get("y_max", None)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    y_ticks = cfg.get("y_ticks", None)
    if y_ticks:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{ytick:.1f}" for ytick in y_ticks], fontsize=12)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # No default grid; we draw only specified ones.
    ax.grid(False)

    # ---- Custom grid lines ----
    if use_log_axis:
        x_grid_exp = cfg.get("x_grid_exp", [0.0, 1.0, 2.0])
        for xg in x_grid_exp:
            ax.axvline(
                xg,
                linestyle=grid_ls,
                linewidth=grid_width,
                color=grid_color
            )

    y_grid = cfg.get("y_grid", [32.0, 33.0])
    for yg in y_grid:
        ax.axhline(
            yg,
            linestyle=grid_ls,
            linewidth=grid_width,
            color=grid_color
        )

    # ---- Boundary lines (spines) ----
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(border_width)

    ax.tick_params(which='both', direction='in')

# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot RD curves from embedded config (no JSON file)."
    )
    ap.add_argument("--out", default="", help="Output image path (png/pdf/svg).")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI for saving.")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not display window; useful on headless.")
    ap.add_argument("--log-scale", action="store_true",
                    help="Use custom log10(x) axis described in the paper.")
    ap.add_argument("--empty-out", default="",
                    help="Optional path for an 'empty' plot with same axes/grids but no data.")
    return ap.parse_args()

# ============================================================
# Main plotting functions
# ============================================================
def plot_with_data(cfg: Dict, args):
    metric_key = cfg.get("metric_key", "metric")
    show_markers = bool(cfg.get("show_markers", True))
    methods = cfg.get("methods", [])
    if not isinstance(methods, list) or len(methods) == 0:
        raise RuntimeError("No methods specified in config.")

    fig, ax = create_figure_and_axes(cfg)

    use_log_axis = bool(args.log_scale)
    setup_axes(ax, cfg, use_log_axis=use_log_axis)

    for i, method in enumerate(methods):
        name = method.get("name", f"Method {i+1}")
        points = method.get("points", [])
        if not points:
            continue

        rates, metrics = extract_xy(points, metric_key)

        if use_log_axis:
            mask = rates > 0
            if not np.all(mask):
                rates = rates[mask]
                metrics = metrics[mask]
            xvals = np.log10(rates)   # log axis from 10^-0.5 to 10^3
        else:
            xvals = rates

        color     = method.get("color", None)
        marker    = method.get("marker", None)
        linestyle = method.get("linestyle", None)

        if color is None or marker is None or linestyle is None:
            ac, am, als = auto_style(i)
            color     = color     if color     is not None else ac
            marker    = marker    if marker    is not None else am
            linestyle = linestyle if linestyle is not None else ('-' if len(xvals) > 1 else 'None')

        # No legend labels (you'll annotate in PowerPoint)
        if len(xvals) == 1:
            ax.scatter(xvals, metrics, color=color, marker=marker, zorder=3)
        else:
            ax.plot(
                xvals, metrics,
                color=color,
                marker=(marker if show_markers else None),
                linestyle=linestyle,
                linewidth=1.2,
                markersize=6,
            )

        r_str = ", ".join(f"{r:.3g}" for r in rates)
        m_str = ", ".join(f"{m:.3g}" for m in metrics)
        print(f"[{name}] rates(MB): [{r_str}]  {cfg.get('y_label','Metric')}: [{m_str}]")

    # Do NOT call tight_layout, it would override our manual axes width
    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure with data to {args.out}")
    if not args.no_show and not args.out:
        plt.show()

def plot_empty(cfg: Dict, args):
    if not args.empty_out:
        return

    fig, ax = create_figure_and_axes(cfg)
    use_log_axis = bool(args.log_scale)
    setup_axes(ax, cfg, use_log_axis=use_log_axis)

    fig.savefig(args.empty_out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved EMPTY figure (no data) to {args.empty_out}")

def main():
    args = parse_args()
    cfg = CFG
    plot_with_data(cfg, args)
    plot_empty(cfg, args)

if __name__ == "__main__":
    main()
