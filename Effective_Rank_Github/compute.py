"""
compute.py  —  Effective-rank analysis: computation script
===========================================================
Configure the USER CONFIGURATION section, then run:

    python compute.py

This script loads channel datasets, computes effective rank, Wasserstein
distances, and CDF data, and saves everything as .npz files under OUTPUT_DIR.

Once results are saved, use the plotting scripts to visualise without
re-running the heavy computation:
    python plot_wasserstein_bar.py
    python plot_cdf.py

Output files saved to OUTPUT_DIR
---------------------------------
effective_rank_values.npz  — per-sample effective-rank for every dataset
wasserstein_data.npz       — W1(reference, each other dataset)
cdf_data.npz               — empirical CDF (x, y) for every dataset
sv_spectrum.npz            — mean normalised singular-value spectrum
summary.csv                — descriptive statistics table
"""

from pathlib import Path
import numpy as np
import pandas as pd

from effective_rank_core import (
    prepare_channel,
    compute_effective_rank,
    summarize_effective_rank,
    compute_wasserstein,
    cdf_xy,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         USER CONFIGURATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── UPA antenna dimensions ────────────────────────────────────────────────────
#    UE side : Nrx_x × Nrx_y  →  Nr = Nrx_x * Nrx_y receive antennas
#    BS side : Ntx_x × Ntx_y  →  Nt = Ntx_x * Ntx_y transmit antennas
ANTENNA = dict(Nrx_x=2, Nrx_y=2, Ntx_x=4, Ntx_y=8)

# ── Effective-rank options ────────────────────────────────────────────────────
USE_POWER_SV = True   # True  = use σ² weights  (recommended)
                       # False = use σ  weights

# ── Matched-subset size for Wasserstein comparison ────────────────────────────
# When comparing distributions of different sizes, this limits both sides to
# N_CLOSEST matched samples before computing W1, which avoids a trivial
# advantage for the dataset with more samples.
#
# Set to None to compute W1 on all available finite samples (no selection).
N_CLOSEST = None

# ── Output folder ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("results/my_experiment")

# ── Reference dataset name ────────────────────────────────────────────────────
# This must exactly match one key in DATASETS below.
# Wasserstein distances are computed as W1(reference, every other dataset).
REFERENCE_NAME = "Sionna RT"

# ── Datasets ──────────────────────────────────────────────────────────────────
# Add, remove, or rename entries freely.  Any number of datasets is supported.
#
# Required keys per entry:
#   path  : str  — path to .npz file
#   type  : str  — "raw_real" | "generated_beamspace"
#   key   : str  — array name inside the .npz
#
# Optional keys (raw_real only):
#   freq_dim : int — axis index that contains subcarriers (multi-carrier files)
#   freq_idx : int — which subcarrier to extract (default: middle index)
#
# Dataset types:
#   "raw_real"
#       Antenna-domain channels (Sionna RT, 3GPP).  The script converts them
#       to beamspace automatically using the UPA DFT codebook.
#   "generated_beamspace"
#       Model outputs already in normalised beamspace, shape (N, 2, Nr, Nt)
#       where axis-1 index 0/1 = real/imaginary part.

DATASETS = {
    "Sionna RT": {
        "path": "data/sionna_channels.npz",
        "type": "raw_real",
        "key":  "combined_array",
        # "freq_dim": 5,   # ← uncomment for multi-carrier files
        # "freq_idx": 128,
    },
    "cDDIM": {
        "path": "data/cdim_channels.npz",
        "type": "generated_beamspace",
        "key":  "channels",
    },
    "cFMM": {
        "path": "data/fmm_channels.npz",
        "type": "generated_beamspace",
        "key":  "channels",
    },
    # Add more entries here — the scripts handle any number of datasets.
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              COMPUTATION  (no changes needed below)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _safe_key(name: str) -> str:
    """Convert dataset name to a safe .npz array key."""
    return name.replace(" ", "_").replace("/", "_").replace(".", "p")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if REFERENCE_NAME not in DATASETS:
        raise ValueError(
            f"REFERENCE_NAME '{REFERENCE_NAME}' not found in DATASETS. "
            f"Available: {list(DATASETS.keys())}"
        )

    # ── 1. Load and compute effective rank ───────────────────────────────────
    erank_all  = {}
    sv_norm_all = {}
    print("=" * 60)
    print("Computing effective rank")
    print("=" * 60)

    for name, info in DATASETS.items():
        print(f"\n  [{name}]")
        H = prepare_channel(info, **ANTENNA)
        print(f"    loaded shape : {H.shape}")
        er, sv = compute_effective_rank(H, use_power=USE_POWER_SV)
        erank_all[name]   = er
        sv_norm_all[name] = sv
        stats = summarize_effective_rank(er)
        print(f"    mean={stats['mean']:.4f}  median={stats['median']:.4f}"
              f"  std={stats['std']:.4f}")

    # ── 2. Save per-sample effective-rank values ─────────────────────────────
    er_path = OUTPUT_DIR / "effective_rank_values.npz"
    np.savez_compressed(er_path, **{_safe_key(k): v for k, v in erank_all.items()})
    print(f"\nSaved : {er_path}")

    # ── 3. Save summary CSV ──────────────────────────────────────────────────
    rows = []
    for name, er in erank_all.items():
        row = summarize_effective_rank(er)
        row["dataset"] = name
        rows.append(row)
    cols = ["dataset", "n", "mean", "std", "min", "p05", "p25",
            "median", "p75", "p95", "max"]
    df = pd.DataFrame(rows)[cols]
    csv_path = OUTPUT_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved : {csv_path}")
    print("\n" + df.to_string(index=False))

    # ── 4. Save average singular-value spectrum ──────────────────────────────
    sv_path = OUTPUT_DIR / "sv_spectrum.npz"
    np.savez_compressed(sv_path, **{
        _safe_key(k): np.mean(v, axis=0) for k, v in sv_norm_all.items()
    })
    print(f"\nSaved : {sv_path}")

    # ── 5. Compute and save Wasserstein distances ────────────────────────────
    print("\n" + "=" * 60)
    print(f"Wasserstein distances  (reference: {REFERENCE_NAME})")
    if N_CLOSEST is not None:
        print(f"  Using matched subsets of N_CLOSEST = {N_CLOSEST}")
    else:
        print("  Using all finite samples (N_CLOSEST = None)")
    print("=" * 60)

    ref_er  = erank_all[REFERENCE_NAME]
    w1_dict = {}
    for name, er in erank_all.items():
        if name == REFERENCE_NAME:
            continue
        w1 = compute_wasserstein(ref_er, er, n_select=N_CLOSEST)
        w1_dict[name] = w1
        print(f"  W1({REFERENCE_NAME} vs {name}) = {w1:.6f}")

    w1_path = OUTPUT_DIR / "wasserstein_data.npz"
    np.savez_compressed(w1_path, **{_safe_key(k): v for k, v in w1_dict.items()})
    print(f"\nSaved : {w1_path}")

    # ── 6. Save CDF data ─────────────────────────────────────────────────────
    cdf_dict = {}
    for name, er in erank_all.items():
        x, y = cdf_xy(er)
        cdf_dict[f"{_safe_key(name)}_x"] = x
        cdf_dict[f"{_safe_key(name)}_y"] = y
    cdf_path = OUTPUT_DIR / "cdf_data.npz"
    np.savez_compressed(cdf_path, **cdf_dict)
    print(f"Saved : {cdf_path}")

    print("\nDone.")
    print("  → Run  plot_wasserstein_bar.py  to render the Wasserstein barplot.")
    print("  → Run  plot_cdf.py              to render the CDF plot.")


if __name__ == "__main__":
    main()
