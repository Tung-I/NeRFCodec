#!/usr/bin/env python3
import argparse, json, itertools, sys
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

"""
Usage:
    python plot_rd.py --json nerf_chair.json --out nerf_chair_rd_curve.png --no-show --log-scale
"""

# ------------------------------
# Helpers
# ------------------------------
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
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5']))
    markers = itertools.cycle(['o','s','^','D','v','P','X','*','<','>'])
    linestyles = itertools.cycle(['-','--','-.',':'])
    # advance cycles i steps without storing global state
    c = None; m = None; ls = None
    for _ in range(i+1):
        c = next(colors); m = next(markers); ls = next(linestyles)
    return c, m, ls

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Plot RD curves from JSON with per-point *_rate_mb summation.")
    ap.add_argument("--json", required=True, help="Path to input JSON.")
    ap.add_argument("--out", default="", help="Output image path (png/pdf/svg). If empty, just shows the plot.")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI for saving.")
    ap.add_argument("--no-show", action="store_true", help="Do not display window; useful on headless.")
    ap.add_argument("--log-scale", action="store_true",
                    help="Use log10 scale on the x-axis (rates).")
    return ap.parse_args()

# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()
    with open(args.json, "r") as f:
        cfg = json.load(f)

    exp_name = cfg.get("experiment_name", "Rate–Distortion")
    x_label  = cfg.get("x_label", "Bitrate (MB)")
    y_label  = cfg.get("y_label", "Metric")
    y_limits = cfg.get("y_limits", None)
    x_limits = cfg.get("x_limits", None)
    legend_ncol = int(cfg.get("legend_ncol", 3))
    metric_key  = cfg.get("metric_key", "metric")
    show_markers = bool(cfg.get("show_markers", True))

    methods = cfg.get("methods", [])
    if not isinstance(methods, list) or len(methods) == 0:
        print("No methods specified in JSON.", file=sys.stderr)
        sys.exit(1)

    plt.figure(figsize=(7.0, 4.8))
    ax = plt.gca()

    # If using log-scale, configure axis before plotting (so minor ticks work nicely)
    if args.log_scale:
        ax.set_xscale('log', base=10)
        # Major ticks at 10^k with mathtext labels
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        # Minor ticks at 2..9 * 10^k
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
        ax.tick_params(which='both', direction='in')

    any_nonpos = False

    for i, method in enumerate(methods):
        # print(f"Processing method: {method}")
        name = method.get("name", f"Method {i+1}")
        points = method.get("points", [])
        if not points:
            continue

        rates, metrics = extract_xy(points, metric_key)

        if args.log_scale:
            # guard against non-positive rates
            mask = rates > 0
            if not np.all(mask):
                any_nonpos = True
                rates = rates[mask]
                metrics = metrics[mask]

        color     = method.get("color", None)
        marker    = method.get("marker", None)
        linestyle = method.get("linestyle", None)
        name      = method.get("name", name)

        if color is None or marker is None or linestyle is None:
            ac, am, als = auto_style(i)
            color     = color     if color     is not None else ac
            marker    = marker    if marker    is not None else am
            linestyle = linestyle if linestyle is not None else ('-' if len(rates) > 1 else 'None')

        if len(rates) == 1:
            # single point -> scatter
            plt.scatter(rates, metrics, label=name, color=color, marker=marker, zorder=3)
        else:
            plt.plot(
                rates, metrics,
                label=name,
                color=color,
                marker=(marker if show_markers else None),
                linestyle=linestyle,
                linewidth=1.6,
                markersize=6
            )

        # Print a small summary row (optional)
        r_str = ", ".join(f"{r:.3g}" for r in rates)
        m_str = ", ".join(f"{m:.3g}" for m in metrics)
        print(f"[{name}] rates(MB): [{r_str}]  {y_label}: [{m_str}]")

    # plt.title(exp_name)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    # if x_limits: plt.xlim(x_limits[0], x_limits[1])
    # if y_limits: plt.ylim(y_limits[0], y_limits[1])

    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)

    # Legend at top
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30),
               ncol=min(legend_ncol, len(methods)), frameon=False, fontsize=10)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {args.out}")
    if not args.no_show and not args.out:
        plt.show()

if __name__ == "__main__":
    main()
