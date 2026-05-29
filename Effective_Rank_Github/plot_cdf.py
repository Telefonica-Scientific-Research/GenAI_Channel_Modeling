"""
plot_cdf.py  —  Effective-rank empirical CDF plot
==================================================
Loads pre-computed cdf_data.npz files (produced by compute.py) and renders
overlaid CDF curves for any combination of datasets and experimental folders.

Usage
-----
    python plot_cdf.py

Edit the USER CONFIGURATION section and re-run instantly — no heavy
recomputation is needed.

How keys work
-------------
compute.py saves each dataset's CDF under the key:
    "{dataset_name_with_spaces_replaced_by_underscores}_x"
    "{dataset_name_with_spaces_replaced_by_underscores}_y"

For example, if DATASETS in compute.py contains "Sionna RT" and "cDDIM",
the cdf_data.npz will have keys:
    Sionna_RT_x, Sionna_RT_y, cDDIM_x, cDDIM_y
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         USER CONFIGURATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── CDF curves ────────────────────────────────────────────────────────────────
# Each entry is a tuple: (display_label, result_folder, dataset_key)
#   display_label : label shown in the legend
#   result_folder : Path to folder containing cdf_data.npz
#   dataset_key   : name of the dataset as it appears in the .npz file
#                   (same as compute.py DATASETS key, spaces replaced by _)
#
# Multiple entries can point to the same folder (different datasets from one
# experiment) or to different folders (comparing across experiments).
# Add, remove, or reorder entries freely — any number is supported.

CURVES = [
    # ── 28 GHz scenario ───────────────────────────────────────────────────
    ("Sionna RT (28 GHz)",  Path("results/28GHz_200"),  "Sionna_RT"),
    ("cDDIM (28 GHz)",      Path("results/28GHz_200"),  "cDDIM"),
    ("cFMM (28 GHz)",       Path("results/28GHz_200"),  "cFMM"),
    # ── 3.5 GHz scenario ──────────────────────────────────────────────────
    # ("Sionna RT (3.5 GHz)", Path("results/3p5GHz_200"), "Sionna_RT"),
    # ("cDDIM (3.5 GHz)",     Path("results/3p5GHz_200"), "cDDIM"),
    # ("cFMM (3.5 GHz)",      Path("results/3p5GHz_200"), "cFMM"),
]

# ── Line styles per display label ─────────────────────────────────────────────
# Any label appearing in CURVES that is NOT listed here will use default style.
CURVE_STYLES = {
    "Sionna RT (28 GHz)":  {"color": "#2e7d32", "linestyle": "--", "linewidth": 2.0},
    "cDDIM (28 GHz)":      {"color": "#1565c0", "linestyle": "-",  "linewidth": 1.8},
    "cFMM (28 GHz)":       {"color": "#c62828", "linestyle": "-",  "linewidth": 1.8},
    "Sionna RT (3.5 GHz)": {"color": "#2e7d32", "linestyle": "--", "linewidth": 2.0},
    "cDDIM (3.5 GHz)":     {"color": "#1565c0", "linestyle": "-.",  "linewidth": 1.8},
    "cFMM (3.5 GHz)":      {"color": "#c62828", "linestyle": ":",  "linewidth": 1.8},
}

# ── Figure geometry ────────────────────────────────────────────────────────────
FIG_W, FIG_H = 6.0, 4.5

# ── Font sizes ─────────────────────────────────────────────────────────────────
FONT_LABEL  = 11
FONT_TICK   = 10
FONT_LEGEND = 10

# ── Axis limits (None = automatic) ────────────────────────────────────────────
XLIM = None       # e.g. (1, 8)
YLIM = (0, 1)

# ── Legend position ────────────────────────────────────────────────────────────
LEGEND_LOC = "lower right"

# ── Output files ──────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("results")
OUTPUT_STEM = "effective_rank_cdf"
SAVE_PDF    = True
SAVE_PNG    = True
SHOW_PLOT   = True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              RENDERING  (no changes needed below)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _setup_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset":  "stix",
        "axes.labelsize":    FONT_LABEL,
        "xtick.labelsize":   FONT_TICK,
        "ytick.labelsize":   FONT_TICK,
        "legend.fontsize":   FONT_LEGEND,
        "legend.frameon":    True,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "0.8",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "grid.alpha":        0.6,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def main():
    _setup_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for display_label, folder, dataset_key in CURVES:
        path = Path(folder) / "cdf_data.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"cdf_data.npz not found in {folder}.\n"
                "Run compute.py with that OUTPUT_DIR first."
            )
        d = np.load(path)
        x_key = f"{dataset_key}_x"
        y_key = f"{dataset_key}_y"
        if x_key not in d.files:
            raise KeyError(
                f"Key '{x_key}' not found in {path}.\n"
                f"Available keys: {d.files}"
            )
        x = d[x_key]
        y = d[y_key]
        style = CURVE_STYLES.get(display_label, {})
        ax.plot(x, y, label=display_label, **style)
        print(f"  Plotted: {display_label}  (n={len(x)})")

    ax.set_xlabel("Effective Rank")
    ax.set_ylabel("CDF")
    if XLIM is not None:
        ax.set_xlim(XLIM)
    if YLIM is not None:
        ax.set_ylim(YLIM)
    ax.legend(loc=LEGEND_LOC)
    ax.set_axisbelow(True)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PDF:
        p = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {p}")
    if SAVE_PNG:
        p = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
