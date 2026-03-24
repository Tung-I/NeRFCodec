import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User-editable configuration
# -----------------------------
SCENE_FILES = {
    "chair": "log/grad_ste_chair_jpeg35/version_000/grad_stats.txt",
    "lego": "log/grad_ste_lego_jpeg35/version_000/grad_stats.txt",
    "drums": "log/grad_ste_drums_jpeg35/version_000/grad_stats.txt",
    "ficus": "log/grad_ste_ficus_jpeg35/version_000/grad_stats.txt",
    "hotdog": "log/grad_ste_hotdog_jpeg35/version_000/grad_stats.txt",
    "materials": "log/grad_ste_materials_jpeg35/version_000/grad_stats.txt",
    "mic": "log/grad_ste_mic_jpeg35/version_000/grad_stats.txt",
    "ship": "log/grad_ste_ship_jpeg35/version_000/grad_stats.txt",
}

SCENES_TO_PLOT = ["chair", "lego", "drums", "ficus", "hotdog", "materials", "mic", "ship"]

WINDOWS = {
    "MSE": (10, 8010),
    "grad L2 norm": (10, 8010),
    "grad over param": (10, 8010),
    "grad p99 abs": (10, 8010),
}

USE_LOG_Y = {
    "MSE": False,
    "grad L2 norm": True,
    "grad over param": True,
    "grad p99 abs": True,
}

X_TICKS = [0, 2000, 4000, 6000, 8000]
Y_TICKS = [0.002, 0.006, 0.010, 0.014]

# Moving average window (in number of logged points, not iterations)
# Example: if you log every 10 iterations, window=101 means ~1010 iterations smoothing.
MOVING_AVG_WINDOW = 5  # edit (odd number recommended)

OUTPUT_FIG = "grad_2x2_panels.png"
DPI = 250

PANEL_TO_COL = {
    "MSE": "mse",
    "grad L2 norm": "den_grad_l2",
    "grad over param": "grad_over_param",
    "grad p99 abs": "grad_p99",
}

# -----------------------------
# Helpers
# -----------------------------
def read_grad_stats(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values("it").reset_index(drop=True)

    required = {"it", "mse", "den_grad_l2", "grad_over_param", "grad_p99"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Found: {list(df.columns)}")
    return df

def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    s = pd.Series(y)
    # center=True gives symmetric smoothing; min_periods avoids NaNs at edges
    return s.rolling(window=window, center=True, min_periods=max(1, window // 5)).mean().to_numpy()

def slice_and_shift(df: pd.DataFrame, a: int, b: int, ycol: str):
    sub = df[(df["it"] >= a) & (df["it"] <= b)]
    if sub.empty:
        return None, None
    x = sub["it"].to_numpy() - a
    y = sub[ycol].to_numpy()
    return x, y

# -----------------------------
# Load data
# -----------------------------
scene_data = {}
for scene in SCENES_TO_PLOT:
    if scene not in SCENE_FILES:
        raise KeyError(f"Scene '{scene}' not in SCENE_FILES mapping.")
    scene_data[scene] = read_grad_stats(SCENE_FILES[scene])

palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5","C6","C7"])
scene_to_color = {scene: palette[i % len(palette)] for i, scene in enumerate(SCENES_TO_PLOT)}

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
axes = axes.flatten()

panels = [
    ("MSE", axes[0]),
    ("grad L2 norm", axes[1]),
    ("grad over param", axes[2]),
    ("grad p99 abs", axes[3]),
]

legend_handles, legend_labels = [], []

for title, ax in panels:
    a, b = WINDOWS[title]
    ycol = PANEL_TO_COL[title]

    for scene in SCENES_TO_PLOT:
        df = scene_data[scene]
        x, y = slice_and_shift(df, a, b, ycol)
        if x is None:
            continue

        y_smooth = moving_average(y, MOVING_AVG_WINDOW)

        line, = ax.plot(
            x, y_smooth,
            linewidth=1.8,
            color=scene_to_color[scene],
            label=scene
        )
        if title == "MSE":
            legend_handles.append(line)
            legend_labels.append(scene)
            ax.set_yticks(Y_TICKS)

    # ax.set_title(title, fontsize=16)
    ax.set_xlabel("Training iteration", fontsize=16)
    ax.set_ylabel(title, fontsize=16)
    ax.set_xticks(X_TICKS)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)

    if USE_LOG_Y.get(title, False):
        ax.set_yscale("log")

# Legend inside upper-right of panel A
axes[0].legend(
    legend_handles, legend_labels,
    loc="upper right",
    frameon=True,
    fontsize=12
)

fig.savefig(OUTPUT_FIG, dpi=DPI, bbox_inches="tight")
print(f"Saved: {OUTPUT_FIG}")