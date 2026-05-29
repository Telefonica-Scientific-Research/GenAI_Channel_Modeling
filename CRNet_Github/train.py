"""
CRNet for CSI Compression with Synthetic Data Augmentation
===========================================================
Original CRNet architecture from:
    https://github.com/Kylin9511/CRNet
    W. Liu et al., "CRNet: An Efficient Multi-task Learning Architecture for
    Large-Scale MIMO CSI Feedback", IEEE TCCN, 2022.

Modifications in this repository:
    - Generalized to support both LoS-only and LoS+NLoS propagation conditions
    - Flexible data loading (Sionna RT .npz, 3GPP .npz, beamspace .npz, .npy)
    - Synthetic data augmentation: mix N_real ground-truth + (N_total - N_real) synthetic
    - Result caching to JSON — replot without retraining (see plot_results.py)
    - UPA 2-D DFT beamspace transform (configurable antenna dimensions)

Usage examples
--------------
# LoS-only, augment with cDDIM synthetic data
python train.py \\
    --condition      los \\
    --real_los       /path/to/sionna_los.npz  --real_format sionna_npz \\
    --synth_los      /path/to/cddim_los.npz   --synth_format beamspace_npz \\
    --method_name    cDDIM \\
    --logs_dir Logs_LoS --plots_dir Plots_LoS

# LoS+NLoS, augment with Flow Matching (FMM) synthetic data
python train.py \\
    --condition      losnlos \\
    --real_los       /path/to/sionna_los.npz  --real_format sionna_npz \\
    --real_nlos      /path/to/sionna_nlos.npz --real_format sionna_npz \\
    --synth_los      /path/to/fmm_los.npz     --synth_format beamspace_npz \\
    --synth_nlos     /path/to/fmm_nlos.npz    --synth_format beamspace_npz \\
    --method_name    "Flow Matching" \\
    --logs_dir Logs --plots_dir Plots

# Real-only (Sionna RT) baseline — omit --synth_* arguments
python train.py \\
    --condition   los \\
    --real_los    /path/to/sionna_los.npz  --real_format sionna_npz \\
    --method_name "Sionna RT" \\
    --logs_dir Logs_LoS --plots_dir Plots_LoS

Supported data formats (--real_format / --synth_format)
--------------------------------------------------------
  sionna_npz      Sionna RT output: npz key 'combined_array', shape (N,Nr,1,Nt,1,Nsc,4)
  3gpp_npz        3GPP output:      npz key 'ChanPos',        shape (N,1,Nr,1,Nt,1,Nsc,4)
  beamspace_npz   cDDIM/FMM output: npz key 'channels',       shape (N,2,Nr,Nt) float32 beamspace
  npy_complex     .npy with shape (N, Nr, Nt) complex64 — antenna domain
  npy_stacked     .npy with shape (N, 2, Nr, Nt) float32 — real/imag on axis 1, antenna domain
"""

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL DEFAULTS  (override via CLI)
# ─────────────────────────────────────────────────────────────────────────────
SEED           = 42
VAL_FRAC       = 0.10          # fraction of training set used for validation
N_REAL_SIZES   = [200, 500, 1000, 2000, 5000, 10000]   # sweep points
TOTAL_TRAIN    = 10_000        # fixed total training size for augmented methods
BENCH_REAL     = 10_000        # real samples used for the benchmark
N_TEST         = 1_000         # fixed test set size
SEEDS          = [42, 123, 2024]   # averaged over these seeds

# ─────────────────────────────────────────────────────────────────────────────
# UPA / BEAMSPACE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    """Unitary 2-D DFT codebook for a UPA with Nx×Ny elements.
    Returns (Nx*Ny, Nx*Ny) = kron(F_Nx, F_Ny), where F_N = DFT_N / sqrt(N).
    """
    Fx = np.fft.fft(np.eye(Nx, dtype=np.complex64), axis=0) / np.sqrt(Nx)
    Fy = np.fft.fft(np.eye(Ny, dtype=np.complex64), axis=0) / np.sqrt(Ny)
    return np.kron(Fx, Fy).astype(np.complex64)


def beamspace_to_antenna(Hv: np.ndarray, nrx_x: int, nrx_y: int,
                          ntx_x: int, ntx_y: int) -> np.ndarray:
    """Inverse UPA beamspace transform: H = Ar @ Hv @ At^H.
    Hv: (..., Nr, Nt) complex. Returns same shape in antenna domain.
    """
    Ar = upa_dft_codebook(nrx_x, nrx_y)
    At = upa_dft_codebook(ntx_x, ntx_y)
    return (Ar @ Hv) @ At.conj().T


def antenna_to_beamspace(H: np.ndarray, nrx_x: int, nrx_y: int,
                          ntx_x: int, ntx_y: int) -> np.ndarray:
    """Forward UPA beamspace transform: Hv = Ar^H @ H @ At.
    H: (..., Nr, Nt) complex. Returns same shape in beamspace.
    """
    Ar = upa_dft_codebook(nrx_x, nrx_y)
    At = upa_dft_codebook(ntx_x, ntx_y)
    return (Ar.conj().T @ H) @ At


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_channels(path: str, fmt: str,
                  nrx_x: int = 2, nrx_y: int = 2,
                  ntx_x: int = 4, ntx_y: int = 8) -> np.ndarray:
    """Load a channel dataset and return (N, Nr, Nt) complex64 in antenna domain.

    Args:
        path:  Path to .npz or .npy file.
        fmt:   Format identifier — see module docstring for options.
        nrx_x, nrx_y: UE UPA dimensions (Nr = nrx_x * nrx_y).
        ntx_x, ntx_y: BS UPA dimensions (Nt = ntx_x * ntx_y).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if fmt == "sionna_npz":
        # combined_array: (N, Nr, 1, Nt, 1, Nsc, 4) complex64
        d   = np.load(str(p), allow_pickle=True)
        raw = d["combined_array"].squeeze()   # squeeze singleton dims
        # raw now: (N, Nr, Nt, Nsc, 4) — take mid subcarrier, first component
        while raw.ndim > 5:
            raw = raw.squeeze(axis=2)
        mid = raw.shape[-2] // 2
        H   = raw[:, :, :, mid, 0].astype(np.complex64)
        return H[~np.all(H == 0, axis=(-2, -1))]

    elif fmt == "3gpp_npz":
        # ChanPos: (N, 1, Nr, 1, Nt, 1, Nsc, 4) complex64
        d   = np.load(str(p), allow_pickle=True)
        raw = d["ChanPos"].squeeze()
        while raw.ndim > 5:
            raw = raw.squeeze(axis=1)
        mid = raw.shape[-2] // 2
        H   = raw[:, :, :, mid, 3].astype(np.complex64)
        return H[~np.all(H == 0, axis=(-2, -1))]

    elif fmt == "beamspace_npz":
        # channels: (N, 2, Nr, Nt) float32 — real/imag stacked in beamspace
        d  = np.load(str(p), allow_pickle=True)
        ch = d["channels"]                                       # (N, 2, Nr, Nt)
        Hv = (ch[:, 0] + 1j * ch[:, 1]).astype(np.complex64)   # (N, Nr, Nt) beamspace
        return beamspace_to_antenna(Hv, nrx_x, nrx_y, ntx_x, ntx_y)

    elif fmt == "npy_complex":
        # (N, Nr, Nt) complex64 — already in antenna domain
        return np.load(str(p)).astype(np.complex64)

    elif fmt == "npy_stacked":
        # (N, 2, Nr, Nt) float32 — real/imag on axis 1, antenna domain
        d = np.load(str(p)).astype(np.float32)
        return (d[:, 0] + 1j * d[:, 1]).astype(np.complex64)

    else:
        raise ValueError(
            f"Unknown format '{fmt}'. Choose from: sionna_npz, 3gpp_npz, "
            "beamspace_npz, npy_complex, npy_stacked"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING  (N, Nr, Nt) complex → (N, 2, Nr, Nt) float32 in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def to_tensor(H: np.ndarray) -> torch.Tensor:
    """Per-sample max-amplitude normalization, real/imag stacking, mapping to [0, 1].

    Because the UPA DFT is unitary, NMSE computed on these tensors equals
    the beamspace NMSE: ||Hv_pred - Hv||² / ||Hv||².
    """
    amp = np.abs(H).max(axis=(-2, -1), keepdims=True) + 1e-12
    Hn  = H / amp
    ri  = np.stack([Hn.real, Hn.imag], axis=1).astype(np.float32)
    return torch.from_numpy((ri + 1.0) / 2.0)


def nmse_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    """NMSE in dB. Both tensors are max-normalized in [0, 1].
    Equivalent to beamspace NMSE due to unitary UPA DFT.
    """
    p = pred.float()   * 2.0 - 1.0
    t = target.float() * 2.0 - 1.0
    num = ((p - t) ** 2).sum(dim=list(range(1, t.ndim)))
    den = (t ** 2).sum(dim=list(range(1, t.ndim))) + 1e-12
    return 10.0 * torch.log10((num / den).mean()).item()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class ConvBN(nn.Sequential):
    """Conv2d + BatchNorm2d with auto same-padding."""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        padding = (
            [(k - 1) // 2 for k in kernel_size]
            if not isinstance(kernel_size, int)
            else (kernel_size - 1) // 2
        )
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size,
                               stride=stride, padding=padding, bias=False)),
            ("bn",   nn.BatchNorm2d(out_planes)),
        ]))


class CRBlock(nn.Module):
    """CRBlock: two parallel paths merged with 1×1 conv + residual shortcut."""
    def __init__(self):
        super().__init__()
        self.path1 = nn.Sequential(
            ConvBN(2, 7, 3),          nn.LeakyReLU(0.3, inplace=True),
            ConvBN(7, 7, [1, 9]),     nn.LeakyReLU(0.3, inplace=True),
            ConvBN(7, 7, [3, 1]),
        )
        self.path2 = nn.Sequential(
            ConvBN(2, 7, [1, 5]),     nn.LeakyReLU(0.3, inplace=True),
            ConvBN(7, 7, [3, 1]),
        )
        self.merge = ConvBN(14, 2, 1)
        self.relu  = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, x):
        return self.relu(self.merge(torch.cat([self.path1(x), self.path2(x)], dim=1)) + x)


class CRNet(nn.Module):
    """CRNet autoencoder for (2, Nr, Nt) channel tensors.

    Encoder: dual-path conv → 1×1 merge → FC bottleneck (latent dim = total // reduction)
    Decoder: FC expand → 5×5 conv → 2×CRBlock → sigmoid
    """
    def __init__(self, nr: int = 4, nt: int = 32, reduction: int = 4):
        super().__init__()
        total = 2 * nr * nt
        self.nr, self.nt = nr, nt

        self.enc_path1 = nn.Sequential(
            ConvBN(2, 2, 3),        nn.LeakyReLU(0.3, inplace=True),
            ConvBN(2, 2, [1, 9]),   nn.LeakyReLU(0.3, inplace=True),
            ConvBN(2, 2, [3, 1]),
        )
        self.enc_path2 = ConvBN(2, 2, 3)
        self.enc_merge = nn.Sequential(
            nn.LeakyReLU(0.3, inplace=True),
            ConvBN(4, 2, 1),
            nn.LeakyReLU(0.3, inplace=True),
        )
        self.enc_fc      = nn.Linear(total, total // reduction)
        self.dec_fc      = nn.Linear(total // reduction, total)
        self.dec_feature = nn.Sequential(
            ConvBN(2, 2, 5), nn.LeakyReLU(0.3, inplace=True),
            CRBlock(), CRBlock(),
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        z = self.enc_merge(
            torch.cat([self.enc_path1(x), self.enc_path2(x)], dim=1)
        )
        z = self.enc_fc(z.view(N, -1))
        return self.sigmoid(self.dec_feature(self.dec_fc(z).view(N, 2, self.nr, self.nt)))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_crnet(
    X_train:    torch.Tensor,
    X_val:      torch.Tensor,
    X_test:     torch.Tensor,
    tag:        str,
    log_path:   Path,
    ckpt_path:  Path,
    nr:         int   = 4,
    nt:         int   = 32,
    reduction:  int   = 4,
    epochs:     int   = 500,
    batch_size: int   = 512,
    lr:         float = 1e-3,
    device:     torch.device = None,
    skip_done:  bool  = True,
    extra_meta: dict  = None,
) -> float:
    """Train CRNet and return test NMSE (dB). Results are cached to log_path."""
    if skip_done and log_path.exists():
        d = json.loads(log_path.read_text())
        print(f"  [CACHED] {tag}: {d['test_nmse_db']:.2f} dB")
        return d["test_nmse_db"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loader(ds, shuffle):
        return DataLoader(TensorDataset(ds), batch_size=batch_size,
                          shuffle=shuffle, pin_memory=True, num_workers=2)

    tr_ld = make_loader(X_train, True)
    vl_ld = make_loader(X_val,   False)
    te_ld = make_loader(X_test,  False)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = CRNet(nr=nr, nt=nt, reduction=reduction).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit  = nn.MSELoss()

    best_val, best_state = float("inf"), None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        for (x,) in tr_ld:
            x = x.to(device)
            opt.zero_grad()
            crit(model(x), x).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(x.to(device)), x.to(device)).item()
                     for (x,) in vl_ld) / len(vl_ld)

        if vl < best_val:
            best_val  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        sch.step()

        if ep % 50 == 0:
            print(f"    [{tag}] ep {ep:4d}/{epochs}  "
                  f"val_mse={vl:.6f}  elapsed={time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for (x,) in te_ld:
            preds.append(model(x.to(device)).cpu())
            targets.append(x)
    test_nmse  = nmse_db(torch.cat(preds), torch.cat(targets))
    elapsed    = time.time() - t0
    print(f"  [{tag}] Test NMSE: {test_nmse:.2f} dB  (time: {elapsed:.0f}s)")

    torch.save(best_state, str(ckpt_path))
    meta = {"tag": tag, "n_train": len(X_train), "n_val": len(X_val),
            "n_test": len(X_test), "reduction": reduction, "epochs": epochs,
            "test_nmse_db": test_nmse, "train_time_s": elapsed}
    if extra_meta:
        meta.update(extra_meta)
    log_path.write_text(json.dumps(meta, indent=2))
    return test_nmse


# ─────────────────────────────────────────────────────────────────────────────
# DATA SPLIT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _carve_test(H: np.ndarray, n_test: int,
                rng: np.random.Generator) -> tuple:
    """Return (test_array, pool_indices) — test is carved out first."""
    perm      = rng.permutation(len(H))
    test_idx  = perm[:n_test]
    pool_idx  = perm[n_test:]
    return H[test_idx], pool_idx


def _make_train_val(H: np.ndarray, pool_idx: np.ndarray, n_real: int,
                    rng: np.random.Generator, val_frac: float = VAL_FRAC):
    """Draw n_real from pool, split into train/val tensors."""
    draw      = pool_idx[rng.permutation(len(pool_idx))[:n_real]]
    H_draw    = H[draw]
    H_draw    = H_draw[rng.permutation(len(H_draw))]
    n_val     = max(1, int(len(H_draw) * val_frac))
    return to_tensor(H_draw[n_val:]), to_tensor(H_draw[:n_val])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SWEEP
# ─────────────────────────────────────────────────────────────────────────────
def run_sweep(
    condition:   str,          # "los" or "losnlos"
    method_name: str,          # display name, e.g. "cDDIM"
    H_real_los:  np.ndarray,   # (N, Nr, Nt) complex64  — ground-truth LoS
    H_real_nlos: np.ndarray,   # (N, Nr, Nt) or None    — ground-truth NLoS
    H_synth_los: np.ndarray,   # (N, Nr, Nt) or None    — synthetic LoS
    H_synth_nlos:np.ndarray,   # (N, Nr, Nt) or None    — synthetic NLoS
    log_dir:     Path,
    ckpt_dir:    Path,
    device:      torch.device,
    nr:          int   = 4,
    nt:          int   = 32,
    reduction:   int   = 4,
    epochs:      int   = 500,
    batch_size:  int   = 512,
    skip_done:   bool  = True,
    n_real_sizes: list = None,
    total_train:  int  = TOTAL_TRAIN,
    bench_real:   int  = BENCH_REAL,
    n_test_per_cond: int = None,
) -> dict:
    """Run the full N_real sweep for one method.

    For `condition='los'`:   uses H_real_los (and H_synth_los if provided).
    For `condition='losnlos'`: mixes LoS + NLoS equally; requires H_real_nlos.

    Returns { "Benchmark": nmse_db, method_name: {n_real: nmse_db, ...} }
    """
    if n_real_sizes is None:
        n_real_sizes = N_REAL_SIZES

    is_losnlos   = (condition == "losnlos")
    has_synth    = H_synth_los is not None
    method_tag   = method_name.lower().replace(" ", "_")
    cond_prefix  = condition  # e.g. "los" or "losnlos"

    # Number of test samples per condition-side
    if n_test_per_cond is None:
        n_test_per_cond = N_TEST // (2 if is_losnlos else 1)

    results: dict = {}

    # ── Carve fixed test set (same across all scenarios) ──────────────────
    rng = np.random.default_rng(SEED)
    H_test_los, pool_los = _carve_test(H_real_los, n_test_per_cond, rng)

    if is_losnlos:
        if H_real_nlos is None:
            raise ValueError("--real_nlos is required for condition 'losnlos'")
        H_test_nlos, pool_nlos = _carve_test(H_real_nlos, n_test_per_cond, rng)
        H_test = np.concatenate([H_test_los, H_test_nlos], axis=0)
    else:
        H_test = H_test_los

    X_test = to_tensor(H_test)

    # ── Benchmark ─────────────────────────────────────────────────────────
    bench_log  = log_dir  / f"{cond_prefix}_benchmark.json"
    bench_ckpt = ckpt_dir / f"{cond_prefix}_benchmark.pt"

    if skip_done and bench_log.exists():
        results["Benchmark"] = json.loads(bench_log.read_text())["test_nmse_db"]
        print(f"  [CACHED] Benchmark: {results['Benchmark']:.2f} dB")
    else:
        rng_b = np.random.default_rng(SEED)
        n_bench_half = bench_real // (2 if is_losnlos else 1)

        if is_losnlos:
            X_tr_b, X_vl_b = _make_train_val(
                np.concatenate([H_real_los, H_real_nlos], axis=0),
                np.concatenate([
                    pool_los[rng_b.permutation(len(pool_los))[:n_bench_half]],
                    pool_nlos[rng_b.permutation(len(pool_nlos))[:n_bench_half]] + len(H_real_los)
                ]),
                n_bench_half * 2, rng_b,
            )
        else:
            X_tr_b, X_vl_b = _make_train_val(H_real_los, pool_los, n_bench_half, rng_b)

        print(f"\n  Benchmark: train={len(X_tr_b):,}  val={len(X_vl_b):,}  test={len(X_test):,}")
        bench_nmse = train_crnet(
            X_tr_b, X_vl_b, X_test,
            tag=f"{cond_prefix}_benchmark", log_path=bench_log, ckpt_path=bench_ckpt,
            nr=nr, nt=nt, reduction=reduction, epochs=epochs,
            batch_size=batch_size, device=device, skip_done=skip_done,
            extra_meta={"condition": condition, "method": "benchmark"},
        )
        results["Benchmark"] = bench_nmse

    # ── N_real sweep ───────────────────────────────────────────────────────
    method_res: dict = {}

    for n_real in n_real_sizes:
        tag_base = f"{cond_prefix}_{method_tag}_nreal{n_real}"
        agg_log  = log_dir / f"{tag_base}_avg.json"

        if skip_done and agg_log.exists():
            d = json.loads(agg_log.read_text())
            method_res[n_real] = d["test_nmse_db"]
            print(f"  [CACHED] {tag_base} avg: {d['test_nmse_db']:.2f} dB")
            continue

        seed_vals = []
        for si, seed in enumerate(SEEDS):
            sc_key  = f"{tag_base}_s{si}"
            sc_log  = log_dir  / f"{sc_key}.json"
            sc_ckpt = ckpt_dir / f"{sc_key}.pt"

            if skip_done and sc_log.exists():
                seed_vals.append(json.loads(sc_log.read_text())["test_nmse_db"])
                continue

            rng_s = np.random.default_rng(seed + n_real)

            if not has_synth:
                # Real-only baseline
                if is_losnlos:
                    n_half = n_real // 2
                    idx_l  = pool_los[rng_s.permutation(len(pool_los))[:n_half]]
                    idx_n  = pool_nlos[rng_s.permutation(len(pool_nlos))[:n_half]]
                    H_all  = np.concatenate([H_real_los[idx_l], H_real_nlos[idx_n]])
                else:
                    idx    = pool_los[rng_s.permutation(len(pool_los))[:n_real]]
                    H_all  = H_real_los[idx]
                n_syn = 0
            else:
                # Augmented: real + synthetic
                if is_losnlos:
                    n_half    = n_real // 2
                    n_syn_half = (total_train - n_real) // 2
                    idx_l  = pool_los[rng_s.permutation(len(pool_los))[:n_half]]
                    idx_n  = pool_nlos[rng_s.permutation(len(pool_nlos))[:n_half]]
                    rng_s2 = np.random.default_rng(seed + n_real + 1)
                    syn_l  = H_synth_los[rng_s2.permutation(len(H_synth_los))[:n_syn_half]]
                    syn_n  = (H_synth_nlos[rng_s2.permutation(len(H_synth_nlos))[:n_syn_half]]
                              if H_synth_nlos is not None else syn_l)
                    H_all  = np.concatenate([H_real_los[idx_l], H_real_nlos[idx_n],
                                             syn_l, syn_n])
                    n_syn  = n_syn_half * 2
                else:
                    n_syn  = total_train - n_real
                    idx    = pool_los[rng_s.permutation(len(pool_los))[:n_real]]
                    rng_s2 = np.random.default_rng(seed + n_real + 1)
                    syn    = H_synth_los[rng_s2.permutation(len(H_synth_los))[:n_syn]]
                    H_all  = np.concatenate([H_real_los[idx], syn])

            rng_mix = np.random.default_rng(seed)
            H_all   = H_all[rng_mix.permutation(len(H_all))]
            n_val   = max(1, int(len(H_all) * VAL_FRAC))
            X_train = to_tensor(H_all[n_val:])
            X_val   = to_tensor(H_all[:n_val])

            print(f"\n{'─'*66}")
            print(f"  {condition.upper()} | {method_name} | N_real={n_real} | seed={seed} "
                  f"| real={n_real}  synth={n_syn}")
            print(f"    → train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

            nmse = train_crnet(
                X_train, X_val, X_test,
                tag=sc_key, log_path=sc_log, ckpt_path=sc_ckpt,
                nr=nr, nt=nt, reduction=reduction, epochs=epochs,
                batch_size=batch_size, device=device, skip_done=skip_done,
                extra_meta={"condition": condition, "method": method_name,
                             "n_real": n_real, "seed": seed},
            )
            seed_vals.append(nmse)

        avg_nmse = float(np.mean(seed_vals))
        agg_log.write_text(json.dumps({
            "test_nmse_db": avg_nmse, "per_seed": seed_vals, "seeds": SEEDS,
            "condition": condition, "method": method_name, "n_real_sizes": n_real_sizes,
        }, indent=2))
        print(f"  {tag_base} avg: {avg_nmse:.2f} dB  per-seed: {[f'{v:.2f}' for v in seed_vals]}")
        method_res[n_real] = avg_nmse

    results[method_name] = method_res
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CRNet CSI compression with synthetic data augmentation"
    )
    # Condition
    parser.add_argument("--condition",    choices=["los", "losnlos"], default="los",
                        help="Propagation condition: 'los' or 'losnlos' (default: los)")
    # Data paths — ground truth
    parser.add_argument("--real_los",    required=True,
                        help="Path to real/GT LoS channel data")
    parser.add_argument("--real_nlos",   default=None,
                        help="Path to real/GT NLoS channel data (required for losnlos)")
    parser.add_argument("--real_format", default="sionna_npz",
                        choices=["sionna_npz", "3gpp_npz", "beamspace_npz",
                                 "npy_complex", "npy_stacked"],
                        help="Format of the ground-truth data files (default: sionna_npz)")
    # Data paths — synthetic
    parser.add_argument("--synth_los",   default=None,
                        help="Path to synthetic LoS channel data (omit for real-only baseline)")
    parser.add_argument("--synth_nlos",  default=None,
                        help="Path to synthetic NLoS channel data")
    parser.add_argument("--synth_format", default="beamspace_npz",
                        choices=["sionna_npz", "3gpp_npz", "beamspace_npz",
                                 "npy_complex", "npy_stacked"],
                        help="Format of the synthetic data files (default: beamspace_npz)")
    # Method / experiment config
    parser.add_argument("--method_name", default="Synthetic",
                        help="Display name for this synthetic method (e.g. 'cDDIM')")
    parser.add_argument("--n_real_sizes", nargs="+", type=int, default=N_REAL_SIZES,
                        help="N_real sweep points (default: 200 500 1000 2000 5000 10000)")
    parser.add_argument("--total_train", type=int, default=TOTAL_TRAIN,
                        help="Fixed total training size for augmented methods (default: 10000)")
    # UPA dimensions
    parser.add_argument("--nrx_x", type=int, default=2, help="UE UPA dim x (default: 2)")
    parser.add_argument("--nrx_y", type=int, default=2, help="UE UPA dim y (default: 2)")
    parser.add_argument("--ntx_x", type=int, default=4, help="BS UPA dim x (default: 4)")
    parser.add_argument("--ntx_y", type=int, default=8, help="BS UPA dim y (default: 8)")
    # Training hyper-parameters
    parser.add_argument("--reduction",  type=int, default=4,  help="Compression ratio (default: 4)")
    parser.add_argument("--epochs",     type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device",     default="cuda:0")
    # Output
    parser.add_argument("--logs_dir",   default="Logs")
    parser.add_argument("--plots_dir",  default="Plots")
    parser.add_argument("--no_skip", dest="skip_done", action="store_false",
                        help="Re-train all scenarios even if cached results exist")
    parser.set_defaults(skip_done=True)

    args = parser.parse_args()

    device = torch.device(
        args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
    )
    nr = args.nrx_x * args.nrx_y
    nt = args.ntx_x * args.ntx_y

    log_dir  = Path(args.logs_dir)
    plt_dir  = Path(args.plots_dir)
    ckpt_dir = log_dir / "checkpoints"
    for d in (log_dir, plt_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*66}")
    print(f"  CRNet CSI Compression  |  condition={args.condition.upper()}")
    print(f"  method      : {args.method_name}")
    print(f"  device      : {device}")
    print(f"  reduction   : {args.reduction}×   epochs: {args.epochs}")
    print(f"  UPA (UE)    : {args.nrx_x}×{args.nrx_y}={nr}   UPA (BS): {args.ntx_x}×{args.ntx_y}={nt}")
    print(f"  N_real sweep: {args.n_real_sizes}")
    print(f"  skip cached : {args.skip_done}")
    print(f"{'='*66}\n")

    # ── Load datasets ──────────────────────────────────────────────────────
    print("Loading real (GT) LoS data ...")
    H_real_los = load_channels(args.real_los, args.real_format,
                               args.nrx_x, args.nrx_y, args.ntx_x, args.ntx_y)
    print(f"  → {len(H_real_los):,} LoS samples  shape={H_real_los.shape[1:]}")

    H_real_nlos = None
    if args.real_nlos:
        print("Loading real (GT) NLoS data ...")
        H_real_nlos = load_channels(args.real_nlos, args.real_format,
                                    args.nrx_x, args.nrx_y, args.ntx_x, args.ntx_y)
        print(f"  → {len(H_real_nlos):,} NLoS samples")

    H_synth_los, H_synth_nlos = None, None
    if args.synth_los:
        print("Loading synthetic LoS data ...")
        H_synth_los = load_channels(args.synth_los, args.synth_format,
                                    args.nrx_x, args.nrx_y, args.ntx_x, args.ntx_y)
        print(f"  → {len(H_synth_los):,} synthetic LoS samples")
    if args.synth_nlos:
        print("Loading synthetic NLoS data ...")
        H_synth_nlos = load_channels(args.synth_nlos, args.synth_format,
                                     args.nrx_x, args.nrx_y, args.ntx_x, args.ntx_y)
        print(f"  → {len(H_synth_nlos):,} synthetic NLoS samples")

    # ── Run sweep ──────────────────────────────────────────────────────────
    results = run_sweep(
        condition=args.condition,
        method_name=args.method_name,
        H_real_los=H_real_los,
        H_real_nlos=H_real_nlos,
        H_synth_los=H_synth_los,
        H_synth_nlos=H_synth_nlos,
        log_dir=log_dir, ckpt_dir=ckpt_dir, device=device,
        nr=nr, nt=nt, reduction=args.reduction,
        epochs=args.epochs, batch_size=args.batch_size,
        skip_done=args.skip_done,
        n_real_sizes=args.n_real_sizes,
        total_train=args.total_train,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    bench = results.get("Benchmark")
    if bench is not None:
        print(f"  {'Benchmark':<48} {bench:>8.2f} dB")
    for n_real, nmse in sorted(results.get(args.method_name, {}).items()):
        label = f"{args.method_name} (N_real={n_real})"
        print(f"  {label:<48} {nmse:>8.2f} dB")

    print(f"\nDone. Logs → {log_dir}")
    print("Run  python plot_results.py --logs_dir {logs_dir}  to generate plots.")


if __name__ == "__main__":
    main()
