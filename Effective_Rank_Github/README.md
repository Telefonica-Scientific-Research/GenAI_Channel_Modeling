# Effective Rank Analysis for UPA MIMO Channels

Tools for computing and visualising the effective rank of UPA MIMO wireless channels, and comparing generative model outputs against a ground-truth reference using the Wasserstein-1 distance.

## Files

| File | Description |
|---|---|
| `effective_rank_core.py` | Core library (DFT/beamspace, channel loading, effective rank, Wasserstein) |
| `compute.py` | Run-once computation script — saves results as `.npz` files |
| `plot_wasserstein_bar.py` | Grouped Wasserstein barplot from saved results |
| `plot_cdf.py` | Empirical CDF plot from saved results |

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure and run computation**

Edit the `USER CONFIGURATION` section in `compute.py`:
- `ANTENNA` — UPA dimensions (Nrx_x, Nrx_y, Ntx_x, Ntx_y)
- `DATASETS` — paths and types of your channel files
- `REFERENCE_NAME` — which dataset is the ground truth
- `OUTPUT_DIR` — where to save results
- `N_CLOSEST` — number of matched samples for Wasserstein (`None` = use all)

Then run:
```bash
python compute.py
```

**3. Plot results** (fast, no recomputation)
```bash
python plot_wasserstein_bar.py
python plot_cdf.py
```

## Supported Channel Formats

- **`raw_real`** — antenna-domain `.npz` (Sionna RT, 3GPP). Automatically converted to beamspace via UPA DFT codebook.
- **`generated_beamspace`** — model output `.npz` with shape `(N, 2, Nr, Nt)` where axis-1 holds real/imaginary parts.
