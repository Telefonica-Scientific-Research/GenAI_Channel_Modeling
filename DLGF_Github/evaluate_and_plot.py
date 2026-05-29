#!/usr/bin/env python
# =============================================================================
#  evaluate_and_plot.py  —  DL-GF Evaluation & Result Plotting
#
#  Based on the original DLGF repository:
#      https://github.com/YuqiangHeng/DLGF
#
#  Modified and extended by Sina Fazel.
#
#  This script:
#    1. Loads one or more trained DL-GF models (produced by train.py).
#    2. Evaluates each model on a specified test dataset (e.g. a held-out
#       ray-tracing dataset) and records average SNR vs. num_probing_beam.
#    3. Computes two absolute upper-bound baselines on the same test data:
#         - MRT + MRC (matched filter, genie-aided beamforming upper bound)
#         - Genie-aided DFT codebook (best beam from DFT codebook)
#    4. Caches all results as a single .npz file (so re-plotting is instant).
#    5. Draws and saves a publication-quality SNR vs. N_probe figure
#       (PDF + EPS + PNG).
#
#  Usage example
#  -------------
#  Evaluate two models against a ray-tracing test set and plot:
#
#      python evaluate_and_plot.py \
#          --model_dir Saved_Models/ \
#          --model_tag  "ds1-RayTracing_n200_rawTrue_npb{NPB}_TxRx_diagonal_FBNone_MLP_BF_loss_noise-94.0dBm_meas16.0_seed7_TX32_RX4" \
#                       "ds1-DiffusionModel_n10000_rawFalse_npb{NPB}_TxRx_diagonal_FBNone_MLP_BF_loss_noise-94.0dBm_meas16.0_seed7_TX32_RX4" \
#          --num_probing_beam 2 4 8 12 16 20 24 28 \
#          --test_ds_path /path/to/raytracing_test_channels.npz \
#          --test_ds_type RayTracing \
#          --curve_label "RT (200 samples)" "cDDIM (10 000 samples)" \
#          --save_cache  Results/cached_results/my_experiment \
#          --save_fig    Results/my_experiment_figure \
#          --Tx_power_dBm 20 --BW 100 --noise_PSD_dB -161
#
#  The {NPB} placeholder in --model_tag is replaced with each value from
#  --num_probing_beam automatically.
#
#  Re-plotting from cache (no model loading):
#
#      python evaluate_and_plot.py --from_cache Results/cached_results/my_experiment.npz \
#          --save_fig Results/my_experiment_figure_v2
# =============================================================================

import os
import argparse
import glob
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split

from DL_utils  import Joint_BF_Autoencoder, eval_model
from beam_utils import pow_2_dB, dB_2_pow


# =============================================================================
# A.  CHANNEL EXTRACTION  (same logic as train.py — kept self-contained)
# =============================================================================

def extract_channels(ds_type, ds_path):
    """
    Load (N, Nr, Nt) complex64 channels from disk.
    Supported ds_type values: RayTracing, Stochastic3GPP,
    DiffusionModel, FlowMatching, AntennaDomain.
    """
    data = np.load(ds_path, allow_pickle=True)

    if ds_type == 'RayTracing':
        raw    = data['combined_array'].squeeze(axis=(2, 4))
        mid_sc = raw.shape[3] // 2
        h      = raw[:, :, :, mid_sc, 0].astype(np.complex64)

    elif ds_type == 'Stochastic3GPP':
        raw    = data['ChanPos'].squeeze(axis=(1, 3, 5))
        mid_sc = raw.shape[3] // 2
        h      = raw[:, :, :, mid_sc, 3].astype(np.complex64)

    elif ds_type in ('DiffusionModel', 'FlowMatching'):
        ch  = data['channels']
        Nr, Nt = ch.shape[2], ch.shape[3]
        Hv  = (ch[:, 0] + 1j * ch[:, 1]).astype(np.complex64)
        def _dft1d(n):
            k = np.arange(n)[:, None]; m = np.arange(n)[None, :]
            return np.exp(-1j * 2 * np.pi * k * m / n) / np.sqrt(n)
        Nr1 = Nr2 = int(np.sqrt(Nr))
        Nt1 = int(np.sqrt(Nt)); Nt2 = Nt // Nt1
        Ar = np.kron(_dft1d(Nr1), _dft1d(Nr2)).astype(np.complex64)
        At = np.kron(_dft1d(Nt1), _dft1d(Nt2)).astype(np.complex64)
        h  = ((Ar @ Hv) @ At.conj().T).astype(np.complex64)

    elif ds_type == 'AntennaDomain':
        h = data.astype(np.complex64)

    else:
        raise ValueError(f"Unknown ds_type: {ds_type!r}")

    valid = ~np.all(h.reshape(h.shape[0], -1) == 0, axis=1)
    h     = h[valid]
    print(f"[{ds_type}] {h.shape[0]} valid channels, shape={h.shape}")
    return h


# =============================================================================
# B.  BASELINE COMPUTATION
# =============================================================================

def compute_baselines(h_test, Tx_power_dBm, noise_power_dBm):
    """
    Compute scalar SNR upper bounds (in dB) over the test set.

    Returns a dict with keys:
        'MRT_MRC'   : matched-filter (MRT+MRC) average SNR
        'genie_DFT' : best-beam DFT codebook average SNR
    """
    from beam_utils import UPA_DFT_codebook

    Nr, Nt = h_test.shape[1], h_test.shape[2]
    noise_lin = dB_2_pow(noise_power_dBm - Tx_power_dBm)

    # --- MRT + MRC ---
    # Per-sample BF gain = ‖H‖_F^2  (singular value bound)
    frob_sq = np.sum(np.abs(h_test) ** 2, axis=(1, 2))     # (N,)
    mrt_mrc_snr_lin = frob_sq / noise_lin
    mrt_mrc_snr_db  = float(np.mean(pow_2_dB(mrt_mrc_snr_lin)))

    # --- Genie-aided DFT ---
    Nr1 = Nr2 = int(np.sqrt(Nr))
    Nt1 = int(np.sqrt(Nt)); Nt2 = Nt // Nt1
    Cb_rx = UPA_DFT_codebook(
        n_azimuth=Nr2 * 2, n_elevation=Nr1 * 2,
        n_antenna_azimuth=Nr2, n_antenna_elevation=Nr1, spacing=0.5
    ).T                                                      # (Nr, n_beams_Rx)
    Cb_tx = UPA_DFT_codebook(
        n_azimuth=Nt2 * 2, n_elevation=Nt1 * 2,
        n_antenna_azimuth=Nt2, n_antenna_elevation=Nt1, spacing=0.5
    ).T                                                      # (Nt, n_beams_Tx)

    # Exhaustive search: BF gain for every (Rx beam, Tx beam) pair per sample
    bf_matrix = np.abs(Cb_rx.conj().T @ h_test @ Cb_tx) ** 2   # (N, n_Rx, n_Tx)
    best_gain  = bf_matrix.reshape(h_test.shape[0], -1).max(axis=1)
    dft_snr_db = float(np.mean(pow_2_dB(best_gain / noise_lin)))

    return {
        'MRT_MRC':   mrt_mrc_snr_db,
        'genie_DFT': dft_snr_db,
    }


# =============================================================================
# C.  MODEL EVALUATION
# =============================================================================

def evaluate_model_sweep(model_tag_template, npb_list, model_dir,
                          h_test_raw, norm_factor, split_seed,
                          Tx_power_dBm, noise_power_dBm, measurement_gain,
                          feedback_mode, num_antenna_Tx, num_antenna_Rx):
    """
    Load the model for each npb value and compute average SNR on h_test_raw.

    model_tag_template : str with {NPB} placeholder
    Returns: (npb_array, snr_db_array)
    """
    noise_lin = dB_2_pow(noise_power_dBm - Tx_power_dBm)

    snr_list = []
    for npb in npb_list:
        tag    = model_tag_template.replace('{NPB}', str(npb))
        fpath  = os.path.join(model_dir, tag + '.pt')
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Model not found: {fpath}")

        meas_noise = dB_2_pow(noise_power_dBm - Tx_power_dBm) / measurement_gain

        # Reconstruct the model
        model = Joint_BF_Autoencoder(
            num_antenna_Tx      = num_antenna_Tx,
            num_antenna_Rx      = num_antenna_Rx,
            num_probing_beam_Tx = npb,
            num_probing_beam_Rx = npb,
            noise_power         = meas_noise,
            norm_factor         = norm_factor,
            feedback            = feedback_mode,
        )
        state = torch.load(fpath, map_location='cpu')
        model.load_state_dict(state)
        model.eval()

        # Scale test channels to match training scale
        h_scaled   = (h_test_raw.T / norm_factor).T.astype(np.complex64)
        torch_h    = torch.from_numpy(h_scaled)

        with torch.no_grad():
            bf_gain, _ = eval_model(
                model, torch_h, h_test_raw,
                noise_power=meas_noise,
                prediction_mode='GF',
                feedback_mode=feedback_mode,
            )

        snr_db = float(np.mean(pow_2_dB(bf_gain / noise_lin)))
        snr_list.append(snr_db)
        print(f"  npb={npb:3d}  SNR={snr_db:.2f} dB")

    return np.array(npb_list), np.array(snr_list)


# =============================================================================
# D.  COMMAND-LINE INTERFACE
# =============================================================================
parser = argparse.ArgumentParser(
    description='DL-GF evaluation and plotting',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# ---- Loading from cache (skip evaluation) -----------------------------------
parser.add_argument('--from_cache', type=str, default=None,
                    help='Path to a previously saved .npz cache. '
                         'If set, evaluation is skipped and the file is '
                         'plotted directly.')

# ---- Models to evaluate -----------------------------------------------------
parser.add_argument('--model_dir', type=str, default='Saved_Models/',
                    help='Directory containing .pt model files.')
parser.add_argument('--model_tag', type=str, nargs='+', default=[],
                    help='One tag template per curve (use {NPB} as placeholder '
                         'for num_probing_beam). Example: '
                         '"ds1-RayTracing_n200_rawTrue_npb{NPB}_..."')
parser.add_argument('--num_probing_beam', type=int, nargs='+',
                    default=[2, 4, 8, 12, 16, 20, 24, 28])
parser.add_argument('--curve_label', type=str, nargs='+', default=[],
                    help='Legend label for each curve (same order as --model_tag).')

# ---- Test dataset -----------------------------------------------------------
parser.add_argument('--test_ds_path', type=str, default=None,
                    help='Path to the test dataset file (.npz or .npy).')
parser.add_argument('--test_ds_type', type=str, default='RayTracing',
                    choices=['RayTracing', 'Stochastic3GPP',
                             'DiffusionModel', 'FlowMatching', 'AntennaDomain'],
                    help='Format of the test dataset.')
parser.add_argument('--n_test', type=int, default=None,
                    help='Maximum number of test samples (default: all).')
parser.add_argument('--split_seed', type=int, default=7,
                    help='Must match the seed used during training.')

# ---- System parameters ------------------------------------------------------
parser.add_argument('--Tx_power_dBm',    type=int,   default=20)
parser.add_argument('--BW',              type=float, default=100,
                    help='Bandwidth in MHz.')
parser.add_argument('--noise_PSD_dB',    type=float, default=-161,
                    help='Noise PSD in dBm/Hz.')
parser.add_argument('--measurement_gain', type=float, default=16.0)
parser.add_argument('--feedback_mode',   type=str,   default='diagonal',
                    choices=['diagonal', 'max', 'full'])

# ---- Model architecture (must match training) --------------------------------
parser.add_argument('--num_antenna_Tx', type=int, default=32)
parser.add_argument('--num_antenna_Rx', type=int, default=4)

# ---- Output -----------------------------------------------------------------
parser.add_argument('--save_cache', type=str,
                    default='Results/cached_results/dlgf_results',
                    help='Path stem for the .npz cache (no extension).')
parser.add_argument('--save_fig', type=str,
                    default='Results/dlgf_snr_figure',
                    help='Path stem for saved figures (no extension). '
                         'Saves .pdf, .eps, and .png.')

# ---- Figure style -----------------------------------------------------------
parser.add_argument('--y_min', type=float, default=None)
parser.add_argument('--y_max', type=float, default=None)
parser.add_argument('--fig_width',  type=float, default=10)
parser.add_argument('--fig_height', type=float, default=7)
parser.add_argument('--dpi',        type=int,   default=150)
parser.add_argument('--font_size',  type=int,   default=14)

args = parser.parse_args()

# =============================================================================
# E.  SYSTEM CONSTANTS
# =============================================================================
noise_power_dBm = args.noise_PSD_dB + 10 * np.log10(args.BW * 1e6)

# =============================================================================
# F.  LOAD OR EVALUATE
# =============================================================================
curves   = []
baselines = {}

if args.from_cache:
    # ---- Re-plot from existing cache ----------------------------------------
    cache    = np.load(args.from_cache, allow_pickle=True)
    n_curves = int(cache['n_curves'])
    for i in range(n_curves):
        curves.append({
            'npb':   cache[f'curve{i}_npb'],
            'snr':   cache[f'curve{i}_snr_db'],
            'label': str(cache[f'curve{i}_label']),
        })
    for k in cache.files:
        if k.startswith('bl_'):
            baselines[k] = float(cache[k])
    print(f"Loaded {n_curves} curves from cache: {args.from_cache}")

else:
    # ---- Full evaluation ----------------------------------------------------
    if not args.test_ds_path:
        raise ValueError("--test_ds_path is required when not using --from_cache")

    # Load test channels
    h_all = extract_channels(args.test_ds_type, args.test_ds_path)
    if args.n_test and args.n_test < h_all.shape[0]:
        rng    = np.random.default_rng(args.split_seed)
        chosen = np.sort(rng.choice(h_all.shape[0], args.n_test, replace=False))
        h_all  = h_all[chosen]
    h_test = h_all
    print(f"Test set: {h_test.shape[0]} channels")

    # Baselines
    print("Computing baselines …")
    baselines_raw = compute_baselines(
        h_test, args.Tx_power_dBm, noise_power_dBm
    )
    for k, v in baselines_raw.items():
        baselines[f'bl_{k}'] = v
        print(f"  {k}: {v:.2f} dB")

    # Per-model evaluation
    norm_factor = float(np.max(np.abs(h_test)))

    labels = args.curve_label or [f"Model {i+1}" for i in range(len(args.model_tag))]
    if len(labels) < len(args.model_tag):
        labels += [f"Model {i+1}" for i in range(len(labels), len(args.model_tag))]

    for tag_tmpl, label in zip(args.model_tag, labels):
        print(f"\nEvaluating: {label}")
        npb_arr, snr_arr = evaluate_model_sweep(
            model_tag_template = tag_tmpl,
            npb_list           = args.num_probing_beam,
            model_dir          = args.model_dir,
            h_test_raw         = h_test,
            norm_factor        = norm_factor,
            split_seed         = args.split_seed,
            Tx_power_dBm       = args.Tx_power_dBm,
            noise_power_dBm    = noise_power_dBm,
            measurement_gain   = args.measurement_gain,
            feedback_mode      = args.feedback_mode,
            num_antenna_Tx     = args.num_antenna_Tx,
            num_antenna_Rx     = args.num_antenna_Rx,
        )
        curves.append({'npb': npb_arr, 'snr': snr_arr, 'label': label})

    # Save cache
    os.makedirs(os.path.dirname(os.path.abspath(args.save_cache)), exist_ok=True)
    cache_path = args.save_cache + '.npz'
    save_dict  = {'n_curves': len(curves)}
    for i, c in enumerate(curves):
        save_dict[f'curve{i}_npb']    = c['npb']
        save_dict[f'curve{i}_snr_db'] = c['snr']
        save_dict[f'curve{i}_label']  = c['label']
    save_dict.update(baselines)
    np.savez(cache_path, **save_dict)
    print(f"\nCache saved → {cache_path}")

# =============================================================================
# G.  PLOT
# =============================================================================

# Colour / style cycle for up to 8 curves
_STYLES = [
    dict(color="#d62728", ls="-",  lw=2.0, marker="o", ms=8,  mfc="none"),
    dict(color="#d62728", ls="-",  lw=2.5, marker="o", ms=9),
    dict(color="#8B8000", ls="-.", lw=2.0, marker="^", ms=8),
    dict(color="#ff7f0e", ls="-",  lw=2.0, marker="p", ms=9),
    dict(color="#1f77b4", ls="--", lw=2.5, marker="D", ms=9),
    dict(color="#2ca02c", ls="-.", lw=2.0, marker="s", ms=8),
    dict(color="#9467bd", ls="--", lw=2.0, marker="v", ms=8),
    dict(color="#8c564b", ls=":",  lw=2.0, marker="x", ms=8),
]

plt.rcParams.update({
    'font.family' : 'serif',
    'font.serif'  : ['Times New Roman', 'Times', 'DejaVu Serif'],
    'pdf.fonttype': 42,
    'ps.fonttype' : 42,
})

fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

# Baseline horizontal lines
if 'bl_MRT_MRC' in baselines:
    ax.axhline(baselines['bl_MRT_MRC'], color='black', ls='-', lw=2.5,
               zorder=4, label='MRT + MRC (upper bound)')
if 'bl_genie_DFT' in baselines:
    ax.axhline(baselines['bl_genie_DFT'], color='#444444', ls='--', lw=2.0,
               zorder=4, label='Genie-aided DFT')

# Learned-beam curves
for i, c in enumerate(curves):
    st = _STYLES[i % len(_STYLES)].copy()
    mfc = st.pop('mfc', st['color'])
    ax.plot(
        c['npb'], c['snr'],
        label=c['label'],
        mfc=mfc, zorder=3,
        **st,
    )

# Axes
fs = args.font_size
ax.set_xlabel(r"Number of probing beam pairs  $N_{\mathrm{probe}}$", fontsize=fs)
ax.set_ylabel("Average SNR (dB)", fontsize=fs)

all_npb = sorted({v for c in curves for v in c['npb'].tolist()})
if all_npb:
    ax.set_xticks(all_npb)
    ax.set_xticklabels([str(v) for v in all_npb], fontsize=fs - 4)
ax.tick_params(axis='y', labelsize=fs - 4)

if args.y_min is not None or args.y_max is not None:
    ax.set_ylim(
        args.y_min if args.y_min is not None else ax.get_ylim()[0],
        args.y_max if args.y_max is not None else ax.get_ylim()[1],
    )

ax.grid(True, alpha=0.30, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels_lg = ax.get_legend_handles_labels()
if handles:
    fig.subplots_adjust(bottom=0.38)
    ax.legend(
        handles, labels_lg,
        loc='upper center', bbox_to_anchor=(0.5, -0.22),
        ncol=2, fontsize=fs - 4,
        frameon=True, edgecolor='grey', framealpha=0.92,
    )

plt.tight_layout()

# Save
os.makedirs(os.path.dirname(os.path.abspath(args.save_fig)), exist_ok=True)
stem = args.save_fig

fig.savefig(stem + '.pdf', dpi=args.dpi, bbox_inches='tight')
print(f"Saved: {stem}.pdf")

fig.savefig(stem + '.eps', dpi=args.dpi, bbox_inches='tight')
print(f"Saved: {stem}.eps")

_gs = subprocess.run(
    ['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m',
     f'-r{args.dpi}', f'-sOutputFile={stem}.png', stem + '.eps'],
    capture_output=True,
)
if _gs.returncode == 0:
    print(f"Saved: {stem}.png")
else:
    # Fallback: matplotlib PNG (slightly different rendering)
    fig.savefig(stem + '.png', dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {stem}.png  (via matplotlib)")
