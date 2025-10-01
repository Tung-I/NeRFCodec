#!/usr/bin/env python3
"""
Plot RD curves for NeRFCodec (single point) and STE+JPEG (curve).

Usage:
  python plot.py --mode planes     # x-axis: size_planes only
  python plot.py --mode total      # x-axis: size_planes + size_codec + size_renderer (default)
  python plot.py --out rd_curve.png

The data is embedded below, but you can easily adapt this script to read JSON.
"""

import argparse
import matplotlib.pyplot as plt

# ---- Embedded data (MB, dB) ----
DATA = {
    "NeRFCodec": {
        "PSNR": 34.75,
        "size_planes": 0.052,
        "size_codec": 1.828,
        "size_renderer": 0.0275,
    },
    "Ours-JPEG-Q80": {
        "PSNR": 35.04,
        "size_planes": 1.05,
        "size_codec": 0.0,
        "size_renderer": 0.0275,
    },
    "Ours-JPEG-Q65": {
        "PSNR": 34.35,
        "size_planes": 0.76,
        "size_codec": 0.0,
        "size_renderer": 0.0275,
    },
    "Ours-JPEG-Q50": {
        "PSNR": 33.83,
        "size_planes": 0.60,
        "size_codec": 0.0,
        "size_renderer": 0.0275,
    },
    "Ours-JPEG-Q35": {
        "PSNR": 33.25,
        "size_planes": 0.47,
        "size_codec": 0.0,
        "size_renderer": 0.0275,
    },
    "Ours-JPEG-Q20": {
        "PSNR": 32.12,
        "size_planes": 0.313,
        "size_codec": 0.0,
        "size_renderer": 0.0275,
    },
}

def compute_x(entry: dict, mode: str) -> float:
    """Compute x-axis size in MB for the given mode."""
    planes = float(entry.get("size_planes", 0.0))
    if mode == "planes":
        return planes
    # "total": planes + codec + renderer
    return planes + float(entry.get("size_codec", 0.0)) + float(entry.get("size_renderer", 0.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["planes", "total"], default="total",
                    help="x-axis definition: planes-only or total (planes+codec+renderer)")
    ap.add_argument("--out", type=str, default="rd_curve.png", help="output figure path")
    args = ap.parse_args()

    # Order JPEG points by quality (for a clean curve)
    jpeg_order = ["Ours-JPEG-Q80", "Ours-JPEG-Q65", "Ours-JPEG-Q50", "Ours-JPEG-Q35", "Ours-JPEG-Q20"]

    # Collect JPEG curve points
    x_jpeg = []
    y_jpeg = []
    labels_jpeg = []
    for key in jpeg_order:
        if key not in DATA:  # safety
            continue
        e = DATA[key]
        x = compute_x(e, args.mode)
        y = float(e["PSNR"])
        x_jpeg.append(x)
        y_jpeg.append(y)
        labels_jpeg.append(key.replace("Ours-JPEG-", ""))

    # NeRFCodec point
    nerf = DATA["NeRFCodec"]
    x_nerf = compute_x(nerf, args.mode)
    y_nerf = float(nerf["PSNR"])

    # Plot
    plt.figure(figsize=(6.5, 4.5), dpi=140)
    # JPEG curve (connected)
    plt.plot(x_jpeg, y_jpeg, marker="*", linewidth=1.8, markersize=10, label="Ours (JPEG)")

    # # Annotate JPEG points with quality labels
    # for xi, yi, lab in zip(x_jpeg, y_jpeg, labels_jpeg):
    #     plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(8, 4), fontsize=8)

    # NeRFCodec single point
    plt.scatter([x_nerf], [y_nerf], marker="o", s=28, label="NeRFCodec-Q6")

    # Axes, grid, legend
    xlabel = "Payload size (MB)"
    if args.mode == "planes":
        xlabel += " (planes only)"
    else:
        xlabel += " (planes + codec + renderer)"

    plt.xlabel(xlabel)
    plt.ylabel("PSNR (dB)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved RD curve to {args.out}")

if __name__ == "__main__":
    main()
