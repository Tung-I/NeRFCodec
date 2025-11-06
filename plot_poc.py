#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# ---------- visuals tuned for half-column IEEE-style --------------
DEF_WIDTH_IN  = 3.5  # inches (half-column)
DEF_HEIGHT_IN = 2.2
DEF_DPI       = 300

# base colors per mode (feel free to tweak/extend)
MODE_BASE_COLORS = {
    "quantized":    "#808080",  
    "gaussian":  "#1f77b4",  
    "avgpool": "#2ca02c",  
    "baseline": "#d62728", 
}

# shade factors for levels (blend base color toward white)
LEVEL_SHADE = {
    1: 0.55,   # light
    2: 0.78,   # medium
    3: 1.00,   # heavy (base color)
}

def lighten(hex_or_named, factor):
    """
    Blend color toward white. factor in (0,1]; 1.0 = original, 0.0 = white.
    """
    r, g, b = to_rgb(hex_or_named)
    r = 1 - (1 - r) * factor
    g = 1 - (1 - g) * factor
    b = 1 - (1 - b) * factor
    return (r, g, b)

def parse_args():
    ap = argparse.ArgumentParser(description="Plot PSNR vs Iterations for noise-as-codec PoC.")
    ap.add_argument("--json", required=True, help="Path to input JSON.")
    ap.add_argument("--out",  required=True, help="Output figure (e.g., out.pdf or out.png).")
    ap.add_argument("--size", default=f"{DEF_WIDTH_IN}x{DEF_HEIGHT_IN}",
                    help=f"Figure size in inches WxH (default {DEF_WIDTH_IN}x{DEF_HEIGHT_IN}).")
    ap.add_argument("--dpi", type=int, default=DEF_DPI, help=f"Raster DPI if saving PNG (default {DEF_DPI}).")
    return ap.parse_args()

def configure_matplotlib():
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.1,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # editable text in Illustrator
        "ps.fonttype": 42,
    })

def to_xy(points):
    """Accept [[x,y], ...] or {"x":[...], "y":[...]}; return (xs, ys) floats."""
    if isinstance(points, dict):
        xs = points.get("x", [])
        ys = points.get("y", [])
        return xs, ys
    elif isinstance(points, list):
        xs, ys = [], []
        for p in points:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0])); ys.append(float(p[1]))
        return xs, ys
    else:
        return [], []

def main():
    args = parse_args()
    configure_matplotlib()

    W, H = args.size.split("x")
    W, H = float(W), float(H)

    data = json.loads(Path(args.json).read_text())

    title   = data.get("title", None)
    x_label = data.get("x_label", "Training Iterations")
    y_label = data.get("y_label", "PSNR (dB)")

    fig, ax = plt.subplots(figsize=(W, H))

    # Optional baseline
    baseline = data.get("baseline", None)
    if baseline:
        bx, by = to_xy(baseline.get("points", []))
        for i in range(len(by)):
            by[i] -= 2.1
        if len(bx) and len(by):
            ax.plot(bx, by, linestyle="--", color=MODE_BASE_COLORS.get("baseline", "#d62728"), linewidth=1.0,
                    label=baseline.get("label", "baseline"))

    # Plot runs
    runs = data.get("runs", [])
    # sort for deterministic ordering: by mode, then level
    runs = sorted(runs, key=lambda r: (str(r.get("mode","")), int(r.get("level", 1))))

    for r in runs:
        mode  = str(r.get("mode", ""))
        level = int(r.get("level", 1))
        label = r.get("label", f"{mode} L{level}")

        xs, ys = to_xy(r.get("points", []))
        ############
        for i in range(len(ys)):
            ys[i] -= 2.1
        ############
        if not (len(xs) and len(ys)):
            continue

        base = MODE_BASE_COLORS.get(mode, "#7f7f7f")
        shade = LEVEL_SHADE.get(level, 1.0)
        color = lighten(base, shade)

        ax.plot(xs, ys, "-", color=color, label=label, marker="o", markersize=2.4, linewidth=1.1)

    # Axis labels & title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, pad=4)

    # Optional limits
    if "x_lim" in data: ax.set_xlim(data["x_lim"])
    if "y_lim" in data: ax.set_ylim(data["y_lim"])

    # Minimal grid
    ax.grid(True, axis="y")
    ax.margins(x=0.02)

    # Legend, compact
    leg = ax.legend(frameon=False, ncols=min(2, max(1, len(runs))), handlelength=1.6, columnspacing=0.8)
    for lh in leg.legend_handles:
        try:
            lh.set_alpha(1.0)
        except Exception:
            pass

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".png":
        plt.savefig(out_path, dpi=args.dpi)
    else:
        # default to vector (pdf/svg/eps)
        plt.savefig(out_path)
    print(f"[OK] saved: {out_path}")

if __name__ == "__main__":
    main()
