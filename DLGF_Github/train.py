#!/usr/bin/env python
# =============================================================================
#  train.py  —  DL Grid-Free (DL-GF) Beam Alignment Training
#
#  Based on the original DLGF repository:
#      https://github.com/YuqiangHeng/DLGF
#
#  Modified and extended by Sina Fazel to support:
#      - Multiple channel dataset types (ray-tracing, stochastic 3GPP,
#        diffusion-model generated, flow-matching generated)
#      - Combined multi-dataset training with automatic scale alignment
#        (per-sample Frobenius normalisation)
#      - Fine-tuning from a pretrained model checkpoint
#      - UPA (Uniform Planar Array) antenna configurations
#      - Sweep over multiple num_probing_beam values in one run
#
#  Usage example:
#      python train.py \
#          --ds1 RayTracing \
#          --ds1_path /path/to/your/channels.npz \
#          --raw1 True \
#          --num_probing_beam 2 4 8 12 16 20 24 28 \
#          --nepoch 500 \
#          --model_save_dir Saved_Models/ \
#          --train_hist_save_dir Train_Hist/
#
#  Dataset format (NPZ or NPY):
#      RayTracing / Stochastic3GPP:
#          NPZ with key 'combined_array' or 'ChanPos'
#          Shape: (N, Nr, 1, Nt, 1, n_subcarrier, 4) — complex, with path-loss
#      Synthetic (DiffusionModel, FlowMatching):
#          NPZ with key 'channels'
#          Shape: (N, 2, Nr, Nt) float32 — real/imag split, beamspace normalised
#      Antenna-domain NPY:
#          .npy file, shape (N, Nr, Nt) complex64 — already in antenna domain
#
#  The train/val/test split is 60/20/20. Test indices are saved alongside
#  the model so evaluate_and_plot.py can reproduce the exact split.
# =============================================================================

import os
import argparse
import time

import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from sklearn.model_selection import train_test_split

from DL_utils import Joint_BF_Autoencoder, BF_loss, fit_alt

torch.cuda.empty_cache()


# =============================================================================
# A.  CHANNEL EXTRACTION
#
#  Each branch converts a raw file into (N, Nr, Nt) complex64 in antenna domain.
#  Add your own branch here if you have a different format.
# =============================================================================

def extract_channels(ds_name, ds_path):
    """
    Load channels from disk and return (N, Nr, Nt) complex64 in antenna domain.

    Supported ds_name values
    ------------------------
    'RayTracing'      : Ray-tracing simulator output (e.g. Sionna RT).
                        NPZ with key 'combined_array':
                        shape (N, Nr, 1, Nt, 1, n_sc, 4) — picks centre
                        subcarrier, last axis index 0 is the channel.
    'Stochastic3GPP'  : 3GPP stochastic geometry output (e.g. DeepMIMO 3GPP).
                        NPZ with key 'ChanPos':
                        shape (N, 1, Nr, 1, Nt, 1, n_sc, 4) — centre
                        subcarrier, last axis index 3 is the channel.
    'DiffusionModel'  : Channels from a conditional diffusion model (cDDIM).
                        NPZ with key 'channels':
                        shape (N, 2, Nr, Nt) float32 real/imag in UPA
                        Kronecker-DFT beamspace (normalised).
    'FlowMatching'    : Channels from a conditional flow-matching model (cFMM).
                        Same NPZ format as DiffusionModel.
    'AntennaDomain'   : Channels already in antenna domain stored as .npy.
                        shape (N, Nr, Nt) complex64.
    """
    if ds_name in ('RayTracing',):
        data = np.load(ds_path, allow_pickle=True)
        raw    = data['combined_array'].squeeze(axis=(2, 4))   # (N, Nr, Nt, n_sc, 4)
        mid_sc = raw.shape[3] // 2
        h = raw[:, :, :, mid_sc, 0].astype(np.complex64)

    elif ds_name in ('Stochastic3GPP',):
        data = np.load(ds_path, allow_pickle=True)
        raw    = data['ChanPos'].squeeze(axis=(1, 3, 5))       # (N, Nr, Nt, n_sc, 4)
        mid_sc = raw.shape[3] // 2
        h = raw[:, :, :, mid_sc, 3].astype(np.complex64)

    elif ds_name in ('DiffusionModel', 'FlowMatching'):
        data = np.load(ds_path, allow_pickle=True)
        ch = data['channels']                                  # (N, 2, Nr, Nt) float32

        # Reverse the UPA Kronecker-DFT beamspace storage:
        #   H_antenna = Ar @ Hv @ At^H
        # where Ar = kron(F_Nr1, F_Nr2), At = kron(F_Nt1, F_Nt2)
        # For ULA (1-D) datasets use np.fft.ifft2 instead.
        Nr, Nt = ch.shape[2], ch.shape[3]
        Hv = (ch[:, 0, :, :] + 1j * ch[:, 1, :, :]).astype(np.complex64)

        def _dft1d(n):
            k = np.arange(n)[:, None]; m = np.arange(n)[None, :]
            return np.exp(-1j * 2 * np.pi * k * m / n) / np.sqrt(n)

        Nr1 = Nr2 = int(np.sqrt(Nr))
        Nt1 = int(np.sqrt(Nt));  Nt2 = Nt // Nt1
        Ar = np.kron(_dft1d(Nr1), _dft1d(Nr2)).astype(np.complex64)
        At = np.kron(_dft1d(Nt1), _dft1d(Nt2)).astype(np.complex64)
        h  = (Ar @ Hv) @ At.conj().T
        h  = h.astype(np.complex64)
        print(f"[{ds_name}] Kronecker IDFT applied → antenna domain "
              f"(scale ~{np.median(np.linalg.norm(h.reshape(h.shape[0], -1), axis=1)):.4f})")

    elif ds_name in ('AntennaDomain',):
        # Pre-computed antenna-domain channels stored as .npy
        data = np.load(ds_path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            raise ValueError("AntennaDomain expects a .npy file, not .npz.")
        h = data.astype(np.complex64)

    else:
        raise ValueError(
            f"Unknown dataset name: {ds_name!r}. "
            "Valid choices: RayTracing, Stochastic3GPP, DiffusionModel, "
            "FlowMatching, AntennaDomain"
        )

    # Remove all-zero samples (missing UEs in some simulators)
    valid = ~np.all(h.reshape(h.shape[0], -1) == 0, axis=1)
    h = h[valid]
    print(f"[{ds_name}] {h.shape[0]} valid channels, shape={h.shape}")
    return h


# =============================================================================
# B.  DATASET PREPARATION  (filter + subsample + optional Frobenius normalise)
# =============================================================================

def prepare_channels(ds_name, ds_path, is_raw, n_samples, seed, use_frob_norm):
    """
    Full preparation pipeline for one dataset:
        1. extract_channels
        2. Optional sub-sampling
        3. Optional per-sample Frobenius normalisation

    Parameters
    ----------
    is_raw       : bool  — True if channels carry physical path-loss
    n_samples    : int or None  — sub-sample to this many channels
    seed         : int  — random seed for sub-sampling
    use_frob_norm: bool — normalise each channel to unit Frobenius norm
    """
    h = extract_channels(ds_name, ds_path)

    if n_samples is not None and n_samples < h.shape[0]:
        rng    = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(h.shape[0], n_samples, replace=False))
        h      = h[chosen]
        print(f"[{ds_name}] Sub-sampled to {n_samples} channels.")

    if use_frob_norm:
        frob = np.sqrt(np.sum(np.abs(h) ** 2, axis=(1, 2), keepdims=True))
        frob = np.where(frob == 0, 1.0, frob)
        h    = (h / frob).astype(np.complex64)
        print(f"[{ds_name}] Per-sample Frobenius normalisation applied.")

    print(f"[{ds_name}] Ready: {h.shape[0]} channels, is_raw={is_raw}")
    return h


# =============================================================================
# C.  COMMAND-LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(
    description='DL-GF beam alignment training',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# ---- Primary dataset --------------------------------------------------------
parser.add_argument('--ds1', type=str, required=True,
                    choices=['RayTracing', 'Stochastic3GPP',
                             'DiffusionModel', 'FlowMatching', 'AntennaDomain'],
                    help='Primary dataset type.')
parser.add_argument('--ds1_path', type=str, required=True,
                    help='Path to the primary dataset file (.npz or .npy).')
parser.add_argument('--n_samples1', type=int, default=None,
                    help='Number of samples to draw from ds1 (default: all).')
parser.add_argument('--raw1', type=str, default='auto',
                    choices=['auto', 'True', 'False'],
                    help='"auto" marks RayTracing/Stochastic3GPP as raw, '
                         'DiffusionModel/FlowMatching/AntennaDomain as synthetic.')

# ---- Optional secondary dataset (for combined / transfer-learning training) --
parser.add_argument('--ds2', type=str, default='none',
                    choices=['none', 'RayTracing', 'Stochastic3GPP',
                             'DiffusionModel', 'FlowMatching', 'AntennaDomain'],
                    help='Secondary dataset type (none = single-dataset mode).')
parser.add_argument('--ds2_path', type=str, default=None,
                    help='Path to the secondary dataset file.')
parser.add_argument('--n_samples2', type=int, default=None,
                    help='Number of samples to draw from ds2 (default: all).')
parser.add_argument('--raw2', type=str, default='auto',
                    choices=['auto', 'True', 'False'])

# ---- Probing beams ----------------------------------------------------------
parser.add_argument('--num_probing_beam', type=int, nargs='+', default=[16],
                    help='Probing-beam counts to train sequentially. '
                         'Example: --num_probing_beam 2 4 8 12 16 20 24 28')

# ---- System parameters ------------------------------------------------------
parser.add_argument('--Tx_power_dBm',   type=int,   default=20)
parser.add_argument('--BW',             type=float, default=100,
                    help='Bandwidth in MHz.')
parser.add_argument('--noise_PSD_dB',   type=float, default=-161,
                    help='Noise PSD in dBm/Hz.')
parser.add_argument('--measurement_gain', type=float, default=16.0,
                    help='Spreading gain for probing measurements.')

# ---- Training hyper-parameters ----------------------------------------------
parser.add_argument('--nepoch',       type=int, default=500)
parser.add_argument('--batch_size',   type=int, default=256)
parser.add_argument('--split_seed',   type=int, default=7,
                    help='Random seed for train/val/test split.')

# ---- Model architecture -----------------------------------------------------
parser.add_argument('--feedback_mode', type=str, default='diagonal',
                    choices=['diagonal', 'max', 'full'])
parser.add_argument('--num_feedback', default=None,
                    help='Top-k measurements to feed back (None = all).')
parser.add_argument('--beam_synthesizer', type=str, default='MLP',
                    choices=['MLP', 'CNN'])
parser.add_argument('--learned_probing', type=str, default='TxRx',
                    choices=['TxRx', 'Tx', 'Rx'])

# ---- GPU --------------------------------------------------------------------
parser.add_argument('--gpu', type=int, default=0,
                    help='CUDA device ID. Uses CPU if no GPU is available.')

# ---- Save directories -------------------------------------------------------
parser.add_argument('--model_save_dir',    type=str, default='Saved_Models/')
parser.add_argument('--train_hist_save_dir', type=str, default='Train_Hist/')

args = parser.parse_args()

# =============================================================================
# D.  DERIVED CONSTANTS
# =============================================================================
_RAW_TYPES = {'RayTracing', 'Stochastic3GPP'}

def _is_raw(ds_name, raw_flag):
    if raw_flag == 'auto':
        return ds_name in _RAW_TYPES
    return raw_flag.lower() == 'true'

raw1 = _is_raw(args.ds1, args.raw1)
raw2 = _is_raw(args.ds2, args.raw2) if args.ds2 != 'none' else True

# Frobenius normalisation is needed whenever any dataset is synthetic
frob_norm = not (raw1 and raw2)

tx_power_dBm         = args.Tx_power_dBm
noise_power_dBm      = args.noise_PSD_dB + 10 * np.log10(args.BW * 1e6)
measurement_gain     = args.measurement_gain
meas_noise_power     = (10 ** ((noise_power_dBm - tx_power_dBm) / 10)
                        / measurement_gain)
num_feedback         = (None if args.num_feedback in (None, 'None')
                        else int(args.num_feedback))

device = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
)
print(f"Device: {device}")
print(f"Frobenius normalisation: {frob_norm}")

os.makedirs(args.model_save_dir,      exist_ok=True)
os.makedirs(args.train_hist_save_dir, exist_ok=True)

# =============================================================================
# E.  LOAD & COMBINE DATASETS
#
#  When mixing raw (physical path-loss) and synthetic (normalised beamspace)
#  channels, per-sample Frobenius normalisation is applied to ALL channels so
#  every sample has unit Frobenius norm at training time.  This eliminates the
#  ~800x scale gap between the two dataset types.  Evaluation SNR remains
#  physically meaningful because the raw test channels are used as-is
#  during evaluation (see evaluate_and_plot.py).
# =============================================================================
if frob_norm:
    print("[Frobenius mode] Scale alignment active (mixed dataset types).")
else:
    print("[Physical mode] All datasets are raw — physical scale preserved.")

h1 = prepare_channels(
    args.ds1, args.ds1_path, raw1,
    args.n_samples1, seed=args.split_seed, use_frob_norm=frob_norm,
)
ds1_count = h1.shape[0]

if args.ds2 != 'none':
    h2 = prepare_channels(
        args.ds2, args.ds2_path, raw2,
        args.n_samples2, seed=args.split_seed + 1, use_frob_norm=frob_norm,
    )
    ds2_count = h2.shape[0]
    h_all     = np.concatenate([h1, h2], axis=0)
    print(f"Combined: {ds1_count} ({args.ds1}) + {ds2_count} ({args.ds2}) "
          f"= {h_all.shape[0]} total")
else:
    ds2_count = 0
    h_all     = h1
    print(f"Single dataset: {ds1_count} channels from {args.ds1}")

num_antenna_Rx, num_antenna_Tx = h_all.shape[1], h_all.shape[2]

# =============================================================================
# F.  GLOBAL NORMALISATION  (max-abs, over full combined dataset)
# =============================================================================
norm_factor = float(np.max(np.abs(h_all)))
h_scaled    = (h_all.T / norm_factor).T.astype(np.complex64)
print(f"Global norm_factor = {norm_factor:.6f}")

# =============================================================================
# G.  TRAIN / VAL / TEST SPLIT  (60 / 20 / 20)
# =============================================================================
all_idc = np.arange(h_all.shape[0])
train_idc, tmp_idc = train_test_split(
    all_idc, test_size=0.4, random_state=args.split_seed
)
val_idc, test_idc = train_test_split(
    tmp_idc, test_size=0.5, random_state=args.split_seed
)
print(f"Split → train: {len(train_idc)}, val: {len(val_idc)}, "
      f"test: {len(test_idc)}")

# =============================================================================
# H.  DATASET TAG  (used in saved file names)
# =============================================================================
n1_tag = str(args.n_samples1) if args.n_samples1 is not None else 'all'
if args.ds2 != 'none':
    n2_tag      = str(args.n_samples2) if args.n_samples2 is not None else 'all'
    dataset_tag = (f"ds1-{args.ds1}_n{n1_tag}_raw{raw1}"
                   f"_ds2-{args.ds2}_n{n2_tag}_raw{raw2}")
else:
    dataset_tag = f"ds1-{args.ds1}_n{n1_tag}_raw{raw1}"

print(f"Dataset tag: {dataset_tag}")

# =============================================================================
# I.  SAVE TEST INDICES  (once, before the per-npb loop)
#
#  Stored alongside the model so evaluate_and_plot.py can reconstruct
#  the exact 20 % test set for any trained model.
# =============================================================================
testidx_fname = os.path.join(args.model_save_dir, dataset_tag + "_testidx.npz")
np.savez(
    testidx_fname,
    train_idc        = train_idc,
    val_idc          = val_idc,
    test_idc         = test_idc,
    ds1_name         = args.ds1,
    ds1_count        = ds1_count,
    ds1_is_raw       = raw1,
    ds2_name         = args.ds2,
    ds2_count        = ds2_count,
    ds2_is_raw       = raw2,
    norm_factor      = norm_factor,
    tx_power_dBm     = tx_power_dBm,
    noise_power_dBm  = noise_power_dBm,
    measurement_gain = measurement_gain,
    split_seed       = args.split_seed,
    frob_norm_used   = frob_norm,
)
print(f"Test indices saved → {testidx_fname}")

# =============================================================================
# J.  TRAINING LOOP  (over each requested num_probing_beam)
# =============================================================================
for n_probing_beam in args.num_probing_beam:

    print(f"\n{'='*60}")
    print(f"  Training  num_probing_beam = {n_probing_beam}")
    print(f"{'='*60}")

    model_tag = (
        f"{dataset_tag}"
        f"_npb{n_probing_beam}"
        f"_{args.learned_probing}"
        f"_{args.feedback_mode}"
        f"_FB{num_feedback}"
        f"_{args.beam_synthesizer}"
        f"_BF_loss"
        f"_noise{noise_power_dBm:.1f}dBm"
        f"_meas{measurement_gain}"
        f"_seed{args.split_seed}"
        f"_TX{num_antenna_Tx}_RX{num_antenna_Rx}"
    )

    model_savefname = os.path.join(args.model_save_dir, model_tag + ".pt")
    hist_pfx        = os.path.join(args.train_hist_save_dir, model_tag)

    # ---- DataLoaders --------------------------------------------------------
    torch_h_train = torch.from_numpy(h_scaled[train_idc]).to(device)
    torch_h_val   = torch.from_numpy(h_scaled[val_idc]).to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch_h_train),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch_h_val),
        batch_size=len(val_idc), shuffle=False,
    )

    # ---- Model --------------------------------------------------------------
    model = Joint_BF_Autoencoder(
        num_antenna_Tx      = num_antenna_Tx,
        num_antenna_Rx      = num_antenna_Rx,
        num_probing_beam_Tx = n_probing_beam,
        num_probing_beam_Rx = n_probing_beam,
        noise_power         = meas_noise_power,
        norm_factor         = norm_factor,
        feedback            = args.feedback_mode,
        num_feedback        = num_feedback,
        learned_probing     = args.learned_probing,
        beam_synthesizer    = args.beam_synthesizer,
    ).to(device)

    # ---- Optimiser ----------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True
    )

    # ---- Loss ---------------------------------------------------------------
    loss_fn = BF_loss(
        noise_power_dBm = noise_power_dBm,
        Tx_power_dBm    = tx_power_dBm,
    )

    # ---- Train --------------------------------------------------------------
    t0 = time.time()
    fit_alt(
        model, train_loader, val_loader,
        optimizer, loss_fn, args.nepoch,
        model_savefname=hist_pfx,
        loss='BF_loss',
        device=device,
    )
    print(f"Finished in {(time.time() - t0)/60:.2f} min")

    # ---- Save ---------------------------------------------------------------
    model = model.cpu()
    torch.save(model.state_dict(), model_savefname)
    print(f"Model saved → {model_savefname}")

print("\nAll probing-beam runs complete.")
