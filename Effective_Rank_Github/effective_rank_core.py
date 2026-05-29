"""
effective_rank_core.py
======================
Core library for effective-rank analysis of UPA MIMO wireless channels.

All functions are pure (no global state, no hardcoded paths or parameters).
Import this module from compute.py, plot_wasserstein_bar.py, or plot_cdf.py.

Supported channel formats
--------------------------
"raw_real"
    Raw antenna-domain .npz files (e.g. Sionna RT, 3GPP).
    Expected array shape after optional subcarrier slicing:
        (..., Nr, ..., Nt, ..., 4)
    The last axis of size 4 contains [channel, x, y, z] or [x, y, z, channel]
    (auto-detected via the imaginary-part heuristic).

"generated_beamspace"
    Model output .npz files already in normalised beamspace.
    Expected shape: (N, 2, Nr, Nt)
        axis-1 index 0 = real part
        axis-1 index 1 = imaginary part
"""

import os
import numpy as np
from scipy.stats import wasserstein_distance as _scipy_w1


# ─────────────────────────────────────────────────────────────────────────────
# DFT / Beamspace
# ─────────────────────────────────────────────────────────────────────────────

def dft_matrix(N: int) -> np.ndarray:
    """Return a unitary N×N DFT matrix (complex64)."""
    n = np.arange(N)
    F = np.exp(-1j * 2.0 * np.pi * n[:, None] * n / N) / np.sqrt(N)
    return F.astype(np.complex64)


def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    """
    Kronecker UPA DFT codebook of shape (Nx*Ny, Nx*Ny).

    Using a consistent Kronecker order is sufficient for effective-rank
    analysis because the resulting matrix is unitary.
    """
    return np.kron(dft_matrix(Nx), dft_matrix(Ny)).astype(np.complex64)


def to_beamspace(H: np.ndarray,
                 Nrx_x: int, Nrx_y: int,
                 Ntx_x: int, Ntx_y: int) -> np.ndarray:
    """
    Transform a batch of antenna-domain MIMO matrices to beamspace.

    Parameters
    ----------
    H       : (N, Nr, Nt) complex array in antenna domain.
    Nrx_x/y : RX UPA dimensions  (Nr = Nrx_x * Nrx_y).
    Ntx_x/y : TX UPA dimensions  (Nt = Ntx_x * Ntx_y).

    Returns
    -------
    (N, Nr, Nt) complex array  Hv = Ar^H @ H @ At
    """
    Ar = upa_dft_codebook(Nrx_x, Nrx_y)
    At = upa_dft_codebook(Ntx_x, Ntx_y)
    Nr, Nt = Nrx_x * Nrx_y, Ntx_x * Ntx_y
    if H.shape[1] != Nr or H.shape[2] != Nt:
        raise ValueError(
            f"Expected H shape (N, {Nr}, {Nt}), got {H.shape}."
        )
    return np.matmul(np.matmul(Ar.conj().T, H), At).astype(np.complex64)


def normalize_max_abs(H: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Per-sample max-magnitude normalisation: H / max|H|.

    This scalar operation does NOT change the effective rank.
    It matches the normalisation applied during diffusion-model training.
    """
    m = np.max(np.abs(H), axis=(1, 2), keepdims=True)
    return (H / np.where(m > eps, m, 1.0)).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# Channel loading
# ─────────────────────────────────────────────────────────────────────────────

def _detect_channel_axis(arr: np.ndarray, max_samples: int = 512) -> int:
    """
    Auto-detect which index in the last dimension (size 4) is the channel.

    The heuristic: the channel component has a non-zero imaginary part,
    while position coordinates are real.  Fallback: smallest magnitude
    (channel << position coordinates).
    """
    if arr.shape[-1] != 4:
        raise ValueError(
            f"Auto-detection expects last dimension = 4, got shape {arr.shape}."
        )
    s = arr[:min(max_samples, arr.shape[0])]
    imag_scores = [float(np.mean(np.abs(np.imag(s[..., i])))) for i in range(4)]
    best = int(np.argmax(imag_scores))
    if imag_scores[best] > 1e-12:
        return best
    # Fallback: channel is typically much smaller than position coordinates
    abs_scores = [float(np.mean(np.abs(s[..., i]))) for i in range(4)]
    return int(np.argmin(abs_scores))


def load_raw_channel(path: str, key: str,
                     freq_dim: int = None,
                     freq_idx: int = None) -> np.ndarray:
    """
    Load a raw antenna-domain channel from a .npz file → (N, Nr, Nt).

    Parameters
    ----------
    path     : Path to .npz file.
    key      : Array key inside the .npz.
    freq_dim : (optional) Axis index that holds subcarriers.
    freq_idx : (optional) Which subcarrier to extract (default: middle).

    Supported array shapes (after optional subcarrier slicing):
        (N, 1, Nr, 1, Nt, 1, 1, 4)  — 3GPP ChanPos
        (N, Nr, 1, Nt, 1, 1, 4)     — Sionna combined_array
        and similar — any shape where last dim = 4 and squeezing yields
        (N, Nr, Nt) after extracting the channel component.
    """
    with np.load(path, allow_pickle=True) as f:
        if key not in f.files:
            raise KeyError(f"Key '{key}' not in {path}. Available: {f.files}")
        arr = f[key]

    if freq_dim is not None:
        n_freq = arr.shape[freq_dim]
        idx = freq_idx if freq_idx is not None else n_freq // 2
        arr = np.take(arr, [idx], axis=freq_dim).squeeze(axis=freq_dim)

    ch = _detect_channel_axis(arr)
    H = np.squeeze(arr[..., ch])
    if H.ndim == 2:
        H = H[np.newaxis]
    if H.ndim != 3:
        raise ValueError(
            f"Expected (N, Nr, Nt) after squeezing, got {H.shape}."
        )
    return H.astype(np.complex64)


def load_generated_channel(path: str, key: str) -> np.ndarray:
    """
    Load a generated (diffusion/flow) channel from a .npz file → (N, Nr, Nt).

    Expected .npz array shape: (N, 2, Nr, Nt)
        index 0 along axis-1 = real part
        index 1 along axis-1 = imaginary part
    """
    with np.load(path, allow_pickle=True) as f:
        if key not in f.files:
            raise KeyError(f"Key '{key}' not in {path}. Available: {f.files}")
        arr = f[key]

    if arr.ndim != 4 or arr.shape[1] != 2:
        raise ValueError(
            f"Expected shape (N, 2, Nr, Nt), got {arr.shape}."
        )
    return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)


def prepare_channel(dataset_info: dict,
                    Nrx_x: int, Nrx_y: int,
                    Ntx_x: int, Ntx_y: int) -> np.ndarray:
    """
    Load one dataset and return a normalised beamspace array (N, Nr, Nt).

    dataset_info keys
    -----------------
    path     : str  — path to .npz file
    type     : str  — "raw_real" | "generated_beamspace"
    key      : str  — array key inside .npz
    freq_dim : int  — (raw_real only) axis with subcarriers
    freq_idx : int  — (raw_real only) subcarrier index (default: middle)
    """
    path  = dataset_info["path"]
    dtype = dataset_info["type"]
    key   = dataset_info["key"]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    if dtype == "raw_real":
        H = load_raw_channel(path, key,
                             freq_dim=dataset_info.get("freq_dim"),
                             freq_idx=dataset_info.get("freq_idx"))
        H = to_beamspace(H, Nrx_x, Nrx_y, Ntx_x, Ntx_y)

    elif dtype == "generated_beamspace":
        H = load_generated_channel(path, key)

    else:
        raise ValueError(
            f"Unknown type '{dtype}'. Use 'raw_real' or 'generated_beamspace'."
        )

    return normalize_max_abs(H)


# ─────────────────────────────────────────────────────────────────────────────
# Effective rank
# ─────────────────────────────────────────────────────────────────────────────

def compute_effective_rank(H: np.ndarray,
                           use_power: bool = True,
                           eps: float = 1e-12):
    """
    Entropy-based effective rank for a batch of MIMO matrices.

    Parameters
    ----------
    H         : (N, Nr, Nt) complex array.
    use_power : If True, use σ² (recommended for wireless energy analysis).
                If False, use σ.
    eps       : Numerical floor.

    Returns
    -------
    erank    : (N,) float64 — effective rank per sample (NaN for degenerate).
    sv_norm  : (N, K) float64 — Frobenius-normalised singular values.
    """
    sv = np.linalg.svd(H, compute_uv=False)             # (N, K)
    modes = sv ** 2 if use_power else sv.copy()

    total = np.sum(modes, axis=1, keepdims=True)
    valid = total[:, 0] > eps

    p = np.zeros_like(modes, dtype=np.float64)
    p[valid] = modes[valid] / total[valid]

    p_safe  = np.where(p > eps, p, 1.0)
    entropy = -np.sum(np.where(p > eps, p * np.log(p_safe), 0.0), axis=1)

    erank        = np.exp(entropy)
    erank[~valid] = np.nan

    frob    = np.sqrt(np.sum(sv ** 2, axis=1, keepdims=True))
    sv_norm = sv / np.maximum(frob, eps)

    return erank.astype(np.float64), sv_norm.astype(np.float64)


def summarize_effective_rank(erank: np.ndarray) -> dict:
    """Return a dict of descriptive statistics for an effective-rank array."""
    v = np.asarray(erank, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {
        "n":      len(v),
        "mean":   float(np.mean(v)),
        "std":    float(np.std(v)),
        "min":    float(np.min(v)),
        "p05":    float(np.percentile(v, 5)),
        "p25":    float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "p75":    float(np.percentile(v, 75)),
        "p95":    float(np.percentile(v, 95)),
        "max":    float(np.max(v)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sample selection (optional — used when N_CLOSEST is set)
# ─────────────────────────────────────────────────────────────────────────────

def select_closest(erank_source: np.ndarray,
                   erank_reference: np.ndarray,
                   n_select: int) -> np.ndarray:
    """
    Return indices of the n_select entries in erank_source that are nearest
    (in value) to any entry in erank_reference, using binary-search NN.

    Useful for comparing distributions of different sizes by selecting a
    matched subset.
    """
    src = np.asarray(erank_source,    dtype=np.float64)
    ref = np.asarray(erank_reference, dtype=np.float64)

    ref_sorted = np.sort(ref[np.isfinite(ref)])
    valid = np.isfinite(src)
    dist  = np.full(len(src), np.inf)

    vals = src[valid]
    pos  = np.searchsorted(ref_sorted, vals)
    hi   = np.clip(pos,     0, len(ref_sorted) - 1)
    lo   = np.clip(pos - 1, 0, len(ref_sorted) - 1)
    dist[valid] = np.minimum(np.abs(vals - ref_sorted[hi]),
                             np.abs(vals - ref_sorted[lo]))

    return np.argsort(dist)[:n_select]


# ─────────────────────────────────────────────────────────────────────────────
# Wasserstein distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_wasserstein(erank_ref: np.ndarray,
                        erank_synth: np.ndarray,
                        n_select: int = None) -> float:
    """
    Wasserstein-1 distance between reference and synthetic effective-rank
    distributions.

    Parameters
    ----------
    erank_ref   : Effective-rank values for the ground-truth dataset.
    erank_synth : Effective-rank values for the synthetic dataset.
    n_select    : If given, select n_select matched samples from each
                  distribution before computing W1 (bilateral nearest-
                  neighbour matching).  If None, use all finite samples.

    Returns
    -------
    Scalar W1 distance.
    """
    ref   = np.asarray(erank_ref,   dtype=np.float64)
    synth = np.asarray(erank_synth, dtype=np.float64)
    ref   = ref[np.isfinite(ref)]
    synth = synth[np.isfinite(synth)]

    if n_select is not None:
        s_idx = select_closest(synth, ref,   n_select)
        synth = synth[s_idx]
        r_idx = select_closest(ref,   synth, n_select)
        ref   = ref[r_idx]

    return float(_scipy_w1(ref, synth))


# ─────────────────────────────────────────────────────────────────────────────
# CDF helper
# ─────────────────────────────────────────────────────────────────────────────

def cdf_xy(values: np.ndarray):
    """Return (x, y) arrays for an empirical CDF (NaN values excluded)."""
    v = np.asarray(values, dtype=np.float64)
    v = np.sort(v[np.isfinite(v)])
    return v, np.arange(1, len(v) + 1) / len(v)
