"""
plot_results.py — Standalone plotting for CRNet CSI Compression results
========================================================================
Based on / adapted from: https://github.com/Kylin9511/CRNet

Reads cached JSON logs produced by train.py and generates publication-quality
NMSE vs. N_real curves for all methods found in the logs directory.

Usage
-----
# Plot LoS-only results:
python plot_results.py --logs_dir Logs_LoS --plots_dir Plots_LoS --condition los

# Plot LoS+NLoS results:
python plot_results.py --logs_dir Logs --plots_dir Plots --condition losnlos

# Combine methods with custom colors/labels:
python plot_results.py --logs_dir Logs_LoS \\
    --methods "cDDIM" "Flow Matching" "3GPP Stochastic" "Sionna RT" \\
    --colors  "#1565c0" "#d32f2f" "#546e7a" "#424242" \\
    --linestyles "--" "-" "-." ":"

JSON naming convention (written by train.py)
--------------------------------------------
  {logs_dir}/{condition}_{method_tag}_nreal{N}_avg.json   per (method, N_real)
  {logs_dir}/{condition}_benchmark.json                   full-data reference
where method_tag = method_name.lower().replace(" ", "_")

Tip: Run this script after each train.py call to see incremental progress.
     Already-trained results are cached; only new points are re-trained.
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# STYLE  — edit here to change the look of the plot
# ─────────────────────────────────────────────────────────────────────────────
FONT_LABEL  = 12      # axis label font size
FONT_TICK   = 12      # tick label font size
FONT_LEGEND = 11      # legend font size
GRID_LS     = "--"
GRID_LW     = 0.6
GRID_ALPHA  = 0.6
FIGSIZE     = (6.7, 2.7)   # width, height in inches

# Default color / style cycle (extended automatically if more methods found)
DEFAULT_STYLES = OrderedDict([
    ("cDDIM",            dict(color="#1565c0", ls="--", marker="o", lw=2.0, ms=5)),
    ("Flow Matching",    dict(color="#d32f2f", ls="-",  marker="o", lw=2.0, ms=5)),
    ("3GPP Stochastic",  dict(color="#546e7a", ls="-.", marker="o", lw=2.0, ms=5)),
    ("Sionna RT",        dict(color="#424242", ls=":",  marker="o", lw=2.0, ms=5, mfc="none")),
])

FALLBACK_COLORS = ["#e67e22", "#8e44ad", "#16a085", "#2980b9", "#27ae60"]
FALLBACK_LS     = ["-", "--", "-.", ":", "-"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize":   FONT_LABEL,
        "xtick.labelsize":  FONT_TICK,
        "ytick.labelsize":  FONT_TICK,
        "legend.fontsize":  FONT_LEGEND,
        "legend.frameon":   True,
        "grid.linestyle":   GRID_LS,
        "grid.linewidth":   GRID_LW,
        "grid.alpha":       GRID_ALPHA,
    })


def _style_axes(ax, grid_axis="both"):
    ax.set_axisbelow(True)
    kw = dict(which="major", linestyle=GRID_LS, linewidth=GRID_LW, alpha=GRID_ALPHA)
    if grid_axis in ("both", "x"):
        ax.xaxis.grid(True, **kw)
    if grid_axis in ("both", "y"):
        ax.yaxis.grid(True, **kw)


def _method_tag(name: str) -> str:
    return name.lower().replace(" ", "_")


def _get_style(method_name: str, idx: int, cli_colors: list, cli_ls: list) -> dict:
    """Return plot style dict for this method."""
    if cli_colors and idx < len(cli_colors):
        return dict(color=cli_colors[idx],
                    ls=cli_ls[idx] if cli_ls and idx < len(cli_ls) else FALLBACK_LS[idx % len(FALLBACK_LS)],
                    marker="o", lw=2.0, ms=5)
    if method_name in DEFAULT_STYLES:
        return DEFAULT_STYLES[method_name]
    ci = idx % len(FALLBACK_COLORS)
    return dict(color=FALLBACK_COLORS[ci],
                ls=FALLBACK_LS[ci], marker="o", lw=2.0, ms=5)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def load_results(logs_dir: Path, condition: str,
                 methods: list = None) -> dict:
    """Scan logs_dir for *_avg.json files and assemble results dict.

    Returns:
        { "Benchmark": float or None,
          method_name: { n_real: nmse_db, ... }, ... }
    """
    results = {}

    # Benchmark
    bench_log = logs_dir / f"{condition}_benchmark.json"
    if bench_log.exists():
        results["Benchmark"] = json.loads(bench_log.read_text())["test_nmse_db"]
    else:
        results["Benchmark"] = None

    # Auto-discover methods if not specified
    if methods is None:
        seen_tags = set()
        for f in sorted(logs_dir.glob(f"{condition}_*_nreal*_avg.json")):
            # filename: {condition}_{method_tag}_nreal{N}_avg.json
            stem  = f.stem                    # e.g. los_cddim_nreal500_avg
            parts = stem.split("_nreal")      # ["los_cddim", "500_avg"]
            if len(parts) == 2:
                tag = parts[0][len(condition) + 1:]  # strip "{condition}_"
                seen_tags.add(tag)
        # Recover display name from the JSON (stored by train.py)
        discovered = {}
        for tag in sorted(seen_tags):
            # find any avg file for this tag to read the display name
            sample = next(logs_dir.glob(f"{condition}_{tag}_nreal*_avg.json"), None)
            if sample:
                d = json.loads(sample.read_text())
                display = d.get("method", tag)
                discovered[display] = tag
        methods_to_load = discovered  # {display_name: tag}
    else:
        methods_to_load = {m: _method_tag(m) for m in methods}

    for display_name, tag in methods_to_load.items():
        method_res = {}
        for f in sorted(logs_dir.glob(f"{condition}_{tag}_nreal*_avg.json")):
            stem  = f.stem
            parts = stem.split("_nreal")
            if len(parts) == 2:
                try:
                    n = int(parts[1].replace("_avg", ""))
                    method_res[n] = json.loads(f.read_text())["test_nmse_db"]
                except (ValueError, KeyError):
                    pass
        if method_res:
            results[display_name] = method_res

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_nmse_vs_nreal(
    results:      dict,
    out_path:     Path,
    condition:    str,
    legend_labels: dict = None,   # {method_name: custom legend label}
    cli_colors:   list  = None,
    cli_ls:       list  = None,
    figsize:      tuple = FIGSIZE,
):
    """Generate NMSE vs. N_real plot from a results dict."""
    _set_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Build legend label template
    _R = r"$N_{\mathrm{real}}$ GT"
    _S = r"$(N_{\mathrm{total}}{-}N_{\mathrm{real}})$"
    _default_lbl = {
        "Sionna RT":       rf"{_R} only",
        "3GPP Stochastic": rf"{_S} 3GPP stochastic + {_R}",
        "cDDIM":           rf"{_S} cDDIM + {_R}",
        "Flow Matching":   rf"{_S} cFMM + {_R}",
    }
    if legend_labels:
        _default_lbl.update(legend_labels)

    # Plot curves
    curve_handles = {}
    method_names  = [k for k in results if k != "Benchmark"]
    for idx, name in enumerate(method_names):
        data = results.get(name)
        if not data:
            continue
        ns    = sorted(data.keys())
        nmses = [data[n] for n in ns]
        st    = _get_style(name, idx, cli_colors, cli_ls)
        mfc   = st.get("mfc", st["color"])
        line, = ax.plot(ns, nmses,
                        color=st["color"], ls=st["ls"], lw=st["lw"],
                        marker=st["marker"], ms=st["ms"], mfc=mfc, zorder=3)
        curve_handles[name] = line

    # Benchmark horizontal line
    bench_h    = None
    bench_nmse = results.get("Benchmark")
    if bench_nmse is not None:
        bench_h = ax.axhline(bench_nmse, color="black", ls="-", lw=2.0, zorder=4)

    # Legend (preserve insertion order, benchmark last)
    leg_handles, leg_labels = [], []
    for name in method_names:
        if name in curve_handles:
            leg_handles.append(curve_handles[name])
            leg_labels.append(_default_lbl.get(name, name))
    if bench_h is not None:
        leg_handles.append(bench_h)
        leg_labels.append(r"Full channel reference (10k GT)")

    # All N_real points found across all methods
    all_ns = sorted({n for name in method_names
                     for n in results.get(name, {}).keys()})
    xlabels = [f"{n/1000:.0f}k" if n >= 1000 else f"{n/1000:.1f}k" for n in all_ns]

    cond_str = "LoS + NLoS" if condition == "losnlos" else "LoS only"
    ax.set_xlabel(r"$N_{\mathrm{real}}$  (GT samples in training set)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_xticks(all_ns)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.legend(leg_handles, leg_labels, loc="upper right")
    _style_axes(ax, "both")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[PLOT] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot CRNet NMSE vs N_real from cached JSON logs"
    )
    parser.add_argument("--logs_dir",   default="Logs",
                        help="Directory containing the *_avg.json log files")
    parser.add_argument("--plots_dir",  default="Plots",
                        help="Directory where the output PNG is saved")
    parser.add_argument("--condition",  choices=["los", "losnlos"], default="los",
                        help="Propagation condition prefix used in log filenames")
    parser.add_argument("--out_name",   default=None,
                        help="Output PNG filename (default: nmse_vs_nreal_{condition}.png)")
    parser.add_argument("--methods",    nargs="*", default=None,
                        help="Method display names to include (default: auto-discover all)")
    parser.add_argument("--colors",     nargs="*", default=None,
                        help="Hex color per method (order matches --methods)")
    parser.add_argument("--linestyles", nargs="*", default=None,
                        help="Linestyle per method (e.g. '--' '-' '-.')")
    parser.add_argument("--figsize",    nargs=2, type=float, default=list(FIGSIZE),
                        metavar=("W", "H"),
                        help=f"Figure size in inches (default: {FIGSIZE[0]} {FIGSIZE[1]})")
    args = parser.parse_args()

    logs_dir  = Path(args.logs_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    results = load_results(logs_dir, args.condition, args.methods)

    methods_found = [k for k in results if k != "Benchmark" and results[k]]
    bench         = results.get("Benchmark")
    print(f"Loaded: Benchmark={bench:.2f} dB" if bench else "Loaded: no benchmark found")
    for m in methods_found:
        pts = ", ".join(f"N={n}:{v:.2f}dB" for n, v in sorted(results[m].items()))
        print(f"  {m}: {pts}")

    if not methods_found:
        print("No method results found. Run train.py first.")
        return

    out_name = args.out_name or f"nmse_vs_nreal_{args.condition}.png"
    plot_nmse_vs_nreal(
        results=results,
        out_path=plots_dir / out_name,
        condition=args.condition,
        cli_colors=args.colors,
        cli_ls=args.linestyles,
        figsize=tuple(args.figsize),
    )


if __name__ == "__main__":
    main()
