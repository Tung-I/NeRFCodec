#!/usr/bin/env python3
"""
Usage:
python plot_new_tanks_results.py \
    --log-scale \
    --out tanks_rd_curve.png \
    --empty-out tanks_rd_curve_empty.png \
    --no-show
"""

import argparse
import itertools
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Global experiment configuration (TanksAndTemples)
# ============================================================
TANKS_CFG = {
    "experiment_name": "TanksAndTemples Dataset",
    "x_label": "Size in log (MB)",  # log10-scale axis
    "y_label": "PSNR (dB)",

    # ---- Figure / axes geometry ----
    "fig_width": 10.0,        # inches: full figure width
    "fig_height": 4.0,       # inches: full figure height
    # Fraction of figure width occupied by the axes.
    # Smaller -> ticks (10^-0.5 ... 10^3) closer together.
    "axes_width_frac": 0.48,   # match NeRF figure
    "axes_height_frac": 0.75,  # same vertical extent

    # ---- Custom log10-x axis design ----
    # We plot x_plot = log10(bitrate).
    "x_exp_min": -0.5,                     # corresponds to 10^{-0.5}
    "x_exp_max": 3.0,                      # corresponds to 10^{3}
    "x_ticks_exp": [-0.5, 0.0, 1.0, 2.0, 3.0],
    "x_grid_exp": [0.0, 1.0, 2.0],         # vertical grid lines at 10^0, 10^1, 10^2

    # ---- Custom y axis design ----
    # Data is roughly 27.4–29.1 dB; use a tight range around it.
    "y_min": 27.1,
    "y_max": 29.2,
    "y_ticks": [27.4, 27.8, 28.2, 28.6, 29.0],
    "y_grid":  [27.4, 27.8, 28.2, 28.6, 29.0],

    # ---- Line and border style (grid + boundaries) ----
    "border_color": "0.8",     # boundary (axes spines) color
    "border_width": 0.8,
    "grid_color": "0.8",       # grid line color
    "grid_width": 0.8,
    "grid_linestyle": "-",     # solid gray grid, same as boundaries

    "metric_key": "metric",
    "show_markers": True,

    # ---- Methods and points (TanksAndTemples data) ----
    # Following your standard:
    # - Rate-control baselines: marker 'o'
    # - Non-rate-control baselines: marker 's'
    "methods": [
        {
            "name": "CatRF-JPEG",
            "linestyle": "-",
            "marker": "o",          # rate-control
            "color": "red",
            "points": [
                # { "plane_rate_mb": 0.785, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 29.10 },
                # { "plane_rate_mb": 0.657, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 28.97 },
                { "plane_rate_mb": 0.785, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 28.97 },
                { "plane_rate_mb": 0.637, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 28.67 },
                { "plane_rate_mb": 0.507, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 28.48 },
                { "plane_rate_mb": 0.377, "feat_vec_rate_mb": 0.049, "renderer_rate_mb": 0.042, "metric": 28.19 }
            ]
        },
        {
            "name": "CNC",
            "linestyle": "-",
            "marker": "o",          # rate-control
            "color": "dodgerblue",  # match NeRF figure
            "points": [
                { "model_rate_mb": 1.532, "metric": 28.76 },
                { "model_rate_mb": 1.090, "metric": 28.51 },
                { "model_rate_mb": 0.884, "metric": 28.32 },
                { "model_rate_mb": 0.720, "metric": 28.15 }
            ]
        },
        {
            "name": "NeRFCodec",
            "linestyle": "-",
            "marker": "o",          # rate-control
            "color": "orange",
            "points": [
                { "model_rate_mb": 1.884, "metric": 27.91 },
                { "model_rate_mb": 1.084, "metric": 27.42 }
            ]
        },
        {
            "name": "BiRF",
            "linestyle": "-",
            "marker": "o",          # rate-control
            "color": "brown",
            "points": [
                { "model_rate_mb": 2.9, "metric": 28.62 },
                { "model_rate_mb": 1.5, "metric": 28.44 },
                { "model_rate_mb": 0.8, "metric": 28.02 }
            ]
        },
        {
            "name": "PPNG",
            "linestyle": "-",
            "marker": "o",
            "color": "gray",
            "points": [
                { "model_rate_mb": 32.8, "metric": 27.83 },
                { "model_rate_mb": 2.49, "metric": 27.23 },
            ]
        },
        {
            "name": "ECRF",
            "linestyle": "-",
            "marker": "o",          # rate-control
            "color": "green",
            "points": [
                { "model_rate_mb": 2.5, "metric": 28.41 },
                { "model_rate_mb": 1.4, "metric": 28.31 },
                { "model_rate_mb": 0.8, "metric": 28.14 }
            ]
        },
        {
            "name": "Masked-Wavelet",
            "linestyle": "-",
            "marker": "o",          # rate-control
            "color": "purple",
            "points": [
                { "model_rate_mb": 2.4, "metric": 28.27 },
                { "model_rate_mb": 1.7, "metric": 28.01 },
                { "model_rate_mb": 0.9, "metric": 27.77 }
            ]
        },
        # Non-rate-control baselines (single points / fixed size) -> marker 's'
        {
            "name": "TensoRF-VM",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 74.08, "metric": 29.01 }
            ]
        },
        {
            "name": "Instant-NGP",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 39.5, "metric": 28.85 }
            ]
        },
        {
            "name": "TensoRF-CP",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 4.0, "metric": 27.59 }
            ]
        },
        {
            "name": "TensoRF High",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 6.7, "metric": 28.24 }
            ]
        },
        # {
        #     "name": "VQ-DVGO",
        #     "linestyle": "None",
        #     "marker": "s",
        #     "points": [
        #         { "model_rate_mb": 1.4, "metric": 28.26 }
        #     ]
        # },
        {
            "name": "VQ-TensoRF",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 3.4, "metric": 28.25 }
            ]
        },
        {
            "name": "K-Planes-hybrid",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 400, "metric": 28.02 }
            ]
        },
        {
            "name": "DVGO",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 600, "metric": 28.38 }
            ]
        },
        {
            "name": "Plenoxels",
            "linestyle": "None",
            "marker": "s",
            "points": [
                { "model_rate_mb": 90, "metric": 27.3 }
            ]
        }
    ]
}

# Pick which config to use
CFG = TANKS_CFG

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

    y_grid = cfg.get("y_grid", [28.0, 29.0])
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
        description="Plot RD curves for TanksAndTemples from embedded config."
    )
    ap.add_argument("--out", default="", help="Output image path (png/pdf/svg).")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI for saving.")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not display window; useful on headless.")
    ap.add_argument("--log-scale", action="store_true",
                    help="Use custom log10(x) axis.")
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
