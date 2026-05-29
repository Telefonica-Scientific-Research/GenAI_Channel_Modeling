"""
plot_wasserstein_bar.py  —  Combined Wasserstein-distance barplot
=================================================================
Loads pre-computed wasserstein_data.npz files (produced by compute.py) and
renders a grouped barplot comparing all generative models across all
experimental scenarios (buckets).

Usage
-----
    python plot_wasserstein_bar.py

Edit the USER CONFIGURATION section and re-run instantly — no heavy
recomputation is needed.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         USER CONFIGURATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Buckets: display label → folder containing wasserstein_data.npz ──────────
# Each folder must contain a wasserstein_data.npz produced by compute.py.
# Reorder, add, or comment-out entries freely — any number is supported.
RESULT_FOLDERS = {
    "28 GHz LoS\n(N=200)":        Path("results/28GHz_200"),
    "3.5 GHz LoS\n(N=200)":       Path("results/3p5GHz_LoS_200"),
    "3.5 GHz LoS+NLoS\n(N=200)":  Path("results/3p5GHz_LoS_NLoS_200"),
    "28 GHz LoS\n(N=1000)":       Path("results/28GHz_1000"),
    "3.5 GHz LoS\n(N=1000)":      Path("results/3p5GHz_LoS_1000"),
    "3.5 GHz LoS+NLoS\n(N=1000)": Path("results/3p5GHz_LoS_NLoS_1000"),
}

# ── Model keys and display labels ────────────────────────────────────────────
# Keys must match the dataset names used in compute.py DATASETS (spaces→_),
# excluding the reference dataset.
MODEL_LABELS = {
    "cDDIM": "GT vs. cDDIM",
    "cFMM":  "GT vs. cFMM",
}

# ── Bar colours and hatch patterns ───────────────────────────────────────────
MODEL_STYLES = {
    "cDDIM": {"facecolor": "#1565c0", "hatch": "///", "edgecolor": "#0d3b75"},
    "cFMM":  {"facecolor": "#c62828", "hatch": "...", "edgecolor": "#7f0000"},
}

# ── Bar annotations (value printed above each bar) ───────────────────────────
ANNOTATE  = True
ANNOT_FMT = "{:.2f}"

# ── Figure geometry ────────────────────────────────────────────────────────────
FIG_W, FIG_H = 6.25, 2.65    # width, height in inches
BAR_WIDTH    = 0.28           # width of each individual bar
BAR_GAP      = 0.01           # gap between bars within a group

# ── Font sizes ─────────────────────────────────────────────────────────────────
FONT_LABEL  = 10
FONT_TICK   = 9
FONT_LEGEND = 10
FONT_ANNOT  = 8

# ── Y-axis range (None = automatic) ───────────────────────────────────────────
YLIM = (0, 0.5)

# ── Legend position ────────────────────────────────────────────────────────────
LEGEND_LOC = "upper left"

# ── Output files ──────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("results")
OUTPUT_STEM = "wasserstein_barplot"
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


def _load_wasserstein(folder: Path) -> dict:
    path = folder / "wasserstein_data.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"wasserstein_data.npz not found in {folder}.\n"
            "Run compute.py with that OUTPUT_DIR first."
        )
    return dict(np.load(path))


def main():
    _setup_style()

    bucket_labels = list(RESULT_FOLDERS.keys())
    model_keys    = list(MODEL_LABELS.keys())
    n_buckets     = len(bucket_labels)
    n_models      = len(model_keys)

    # ── Load results ──────────────────────────────────────────────────────────
    print("Loading Wasserstein data:")
    data = {}
    for label, folder in RESULT_FOLDERS.items():
        d = _load_wasserstein(folder)
        data[label] = {k: float(d[k]) for k in model_keys if k in d}
        short = label.replace("\n", " ")
        for k in model_keys:
            if k in data[label]:
                print(f"  {short:28s}  {MODEL_LABELS[k]:16s}  W1={data[label][k]:.6f}")

    # ── Compute bar positions ─────────────────────────────────────────────────
    group_width   = n_models * BAR_WIDTH + (n_models - 1) * BAR_GAP
    group_centers = np.arange(n_buckets, dtype=float)
    offsets = np.array([
        -(group_width / 2) + i * (BAR_WIDTH + BAR_GAP) + BAR_WIDTH / 2
        for i in range(n_models)
    ])

    # ── Draw plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for i, key in enumerate(model_keys):
        values = [data[b].get(key, 0.0) for b in bucket_labels]
        bars = ax.bar(
            group_centers + offsets[i], values, width=BAR_WIDTH,
            facecolor=MODEL_STYLES[key]["facecolor"],
            hatch=MODEL_STYLES[key]["hatch"],
            edgecolor=MODEL_STYLES[key]["edgecolor"],
            label=MODEL_LABELS[key],
            linewidth=0.6,
            zorder=3,
        )
        if ANNOTATE:
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    ANNOT_FMT.format(val),
                    ha="center", va="bottom",
                    fontsize=FONT_ANNOT,
                    fontfamily="serif",
                    fontweight="bold",
                )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Wasserstein Distance")
    ax.legend(loc=LEGEND_LOC)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    if YLIM is not None:
        ax.set_ylim(YLIM)
    fig.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
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
