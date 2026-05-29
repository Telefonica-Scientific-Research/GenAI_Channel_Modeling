"""
Channel Generation with Sionna RT (Ray Tracing)
================================================

Generates large-scale MIMO channel datasets from a Mitsuba scene using
NVIDIA Sionna RT (open-source ray tracing engine).

  https://nvlabs.github.io/sionna/

Supports:
  - Any carrier frequency and antenna configuration
  - LoS / NLoS / all-user filtering
  - Memory-efficient chunked processing for large UE grids
  - Multi-realization mode (independent Monte-Carlo samples per UE)
  - Optional full RT parameter export (AoA, AoD, ZoA, ZoD, delay, CIR)

Output
------
Each processing chunk is saved as a cluster .npz file.
After all chunks finish, the clusters are merged into a single final .npz.

Final OFDM channel array layout
  shape : [num_users, num_rx_ant, num_tx, num_tx_ant, num_steps, num_subcarriers, 3]
  last dim : [complex_channel, x_pos, y_pos]
  (z coordinate is appended during the merge step → final last dim = 4)

Multi-realization mode inserts an extra realization axis:
  shape : [num_users, num_rx_ant, num_tx, num_tx_ant, num_realizations, num_steps, num_subcarriers, 3]

RT parameter arrays (--save_rt_params):
  shape : [num_users, num_rx_ant, num_tx, num_tx_ant, num_paths, 4]
  last dim : [parameter_value, x_pos, y_pos, z_pos]

Usage
-----
  python channel_generation.py \\
      --scene       scene.xml          \\
      --ue_csv      ue_positions.csv   \\
      --frequency   28e9               \\
      --tx_rows 1   --tx_cols 32       \\
      --rx_rows 1   --rx_cols 4        \\
      --los_filter  los                \\
      --output_prefix  channels_28GHz

  # Save full RT parameters too
  python channel_generation.py --scene scene.xml --ue_csv ue_positions.csv \\
      --save_rt_params

  # Multi-realization (20 independent snapshots per user)
  python channel_generation.py --scene scene.xml --ue_csv ue_positions.csv \\
      --num_realizations 20

Input CSV columns (required)
  x, y           : UE horizontal positions [m]
  los_ray_tracing: boolean LoS flag (required when --los_filter != all)

Input CSV columns (optional – TX configuration)
  If --tx_info_csv is provided, the first row must contain:
  azimuth, tilt, ant_height
"""

import os
import gc
import math
import argparse
import zipfile
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import sionna
from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    PathSolver,
    subcarrier_frequencies,
)

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted  # fallback if natsort is not installed


# ---------------------------------------------------------------------------
# GPU / TF setup
# ---------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate MIMO channel data with Sionna RT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Scene & positions ---
    g = p.add_argument_group("Input files")
    g.add_argument("--scene", required=True,
                   help="Path to the Mitsuba scene XML file.")
    g.add_argument("--ue_csv", required=True,
                   help="CSV with UE positions.  Required columns: x, y. "
                        "Optional LoS column: los_ray_tracing (bool).")
    g.add_argument("--tx_info_csv", default=None,
                   help="Optional CSV whose first row contains TX configuration "
                        "columns: azimuth [deg], tilt [deg], ant_height [m]. "
                        "If omitted, use --tx_position / --tx_azimuth / --tx_tilt.")

    # --- TX position (used when --tx_info_csv is not provided) ---
    g2 = p.add_argument_group("TX position (when --tx_info_csv is not used)")
    g2.add_argument("--tx_x", type=float, default=0.0, help="TX x position [m].")
    g2.add_argument("--tx_y", type=float, default=0.0, help="TX y position [m].")
    g2.add_argument("--tx_height", type=float, default=25.0,
                    help="TX antenna height above ground [m].")
    g2.add_argument("--tx_azimuth", type=float, default=0.0,
                    help="TX boresight azimuth [degrees].")
    g2.add_argument("--tx_tilt", type=float, default=-10.0,
                    help="TX mechanical tilt [degrees].  Negative = downtilt.")

    # --- Frequency ---
    g3 = p.add_argument_group("Frequency")
    g3.add_argument("--frequency", type=float, default=28e9,
                    help="Carrier frequency [Hz].  E.g. 28e9 for 28 GHz, 3.5e9 for sub-6 GHz.")

    # --- Antenna arrays ---
    g4 = p.add_argument_group("Antenna arrays")
    g4.add_argument("--tx_rows", type=int, default=1, help="TX array rows.")
    g4.add_argument("--tx_cols", type=int, default=32, help="TX array columns.")
    g4.add_argument("--rx_rows", type=int, default=1, help="RX array rows.")
    g4.add_argument("--rx_cols", type=int, default=4, help="RX array columns.")
    g4.add_argument("--h_spacing", type=float, default=0.5,
                    help="Horizontal antenna spacing [wavelengths].")
    g4.add_argument("--v_spacing", type=float, default=0.5,
                    help="Vertical antenna spacing [wavelengths].")
    g4.add_argument("--antenna_pattern", default="iso",
                    help="Antenna element pattern (e.g. 'iso', 'dipole').")
    g4.add_argument("--polarization", default="V",
                    help="Antenna polarization (e.g. 'V', 'H', 'cross').")

    # --- OFDM ---
    g5 = p.add_argument_group("OFDM / subcarrier config")
    g5.add_argument("--num_subcarriers", type=int, default=1,
                    help="Number of OFDM subcarriers.")
    g5.add_argument("--subcarrier_spacing", type=float, default=60e3,
                    help="OFDM subcarrier spacing [Hz].  E.g. 15e3, 30e3, 60e3, 120e3.")

    # --- Ray tracing ---
    g6 = p.add_argument_group("Ray tracing parameters")
    g6.add_argument("--max_depth", type=int, default=3,
                    help="Maximum ray interaction depth (reflections + diffractions).")
    g6.add_argument("--samples_per_src", type=int, default=5_000_000,
                    help="Number of ray samples launched per source.")
    g6.add_argument("--max_num_paths", type=int, default=1_000_000,
                    help="Maximum number of paths stored per source.")
    g6.add_argument("--los", action="store_true", default=True,
                    help="Include line-of-sight paths.")
    g6.add_argument("--specular_reflection", action="store_true", default=True,
                    help="Include specular reflections.")
    g6.add_argument("--diffraction", action="store_true", default=True,
                    help="Include edge diffraction.")
    g6.add_argument("--diffuse_reflection", action="store_true", default=False,
                    help="Include diffuse / scattered reflections.")
    g6.add_argument("--refraction", action="store_true", default=False,
                    help="Include refraction through surfaces.")
    g6.add_argument("--synthetic_array", action="store_true", default=False,
                    help="Use synthetic array approximation (faster, less accurate).")

    # --- UE filtering ---
    g7 = p.add_argument_group("UE filtering")
    g7.add_argument("--los_filter", choices=["los", "nlos", "all"], default="all",
                    help="Which UEs to process: 'los' = LoS only, 'nlos' = NLoS only, "
                         "'all' = no filtering (requires los_ray_tracing column in CSV for los/nlos).")
    g7.add_argument("--rx_height", type=float, default=1.5,
                    help="UE receiver height above ground [m].")

    # --- Multi-realization ---
    g8 = p.add_argument_group("Multi-realization")
    g8.add_argument("--num_realizations", type=int, default=1,
                    help="Number of independent channel realizations per UE. "
                         "Each uses a different Monte-Carlo seed. "
                         "1 = single realization (standard mode).")
    g8.add_argument("--base_seed", type=int, default=41,
                    help="Base random seed. Seeds used: base_seed, base_seed+1, …")

    # --- RT parameters ---
    g9 = p.add_argument_group("RT parameter export")
    g9.add_argument("--save_rt_params", action="store_true",
                    help="Save AoA, AoD, ZoA, ZoD, delay (tau), and CIR coefficients (a) "
                         "for each path, in addition to the OFDM channel.")

    # --- Processing ---
    g10 = p.add_argument_group("Processing")
    g10.add_argument("--chunk_size", type=int, default=200,
                     help="Number of UEs processed per cluster (memory budget control).")
    g10.add_argument("--output_dir", default=".",
                     help="Directory where output files are written.")
    g10.add_argument("--output_prefix", default="channel_output",
                     help="Prefix for output file names.")
    g10.add_argument("--delete_cluster_files", action="store_true",
                     help="Delete intermediate cluster .npz files after merging.")

    return p


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _tx_orientation_from_degrees(azimuth_deg: float, tilt_deg: float):
    """Convert azimuth / tilt in degrees to Sionna's (alpha, beta, gamma) convention."""
    rad = np.radians([azimuth_deg, tilt_deg, 0.0])
    rad[1] -= np.pi / 2   # Sionna: beta = tilt – 90°
    return rad.tolist()


def _append_xy_complex(ofdm: np.ndarray, user_xy: np.ndarray) -> np.ndarray:
    """Append UE (x, y) positions as the last dimension of a complex OFDM tensor.

    Parameters
    ----------
    ofdm     : complex ndarray, shape [num_rx, ...]
    user_xy  : float32 ndarray, shape [num_rx, 2]

    Returns
    -------
    complex64 ndarray, last dim = [channel_value, x, y]
    """
    num_rx   = ofdm.shape[0]
    n_middle = ofdm.ndim - 1
    user_r   = user_xy.reshape([num_rx] + [1] * n_middle + [2])
    tile_m   = [1] + list(ofdm.shape[1:]) + [1]
    user_t   = np.tile(user_r, tile_m).astype(np.complex64)
    return np.concatenate([ofdm[..., np.newaxis], user_t], axis=-1)


def _append_xyz_float(param: np.ndarray, user_xyz: np.ndarray) -> np.ndarray:
    """Append UE (x, y, z) positions to a real-valued RT parameter tensor.

    Parameters
    ----------
    param    : float ndarray, shape [num_rx, ..., num_paths]
    user_xyz : float32 ndarray, shape [num_rx, 3]

    Returns
    -------
    float32 ndarray, last dim = [param_value, x, y, z]
    """
    num_rx   = param.shape[0]
    n_middle = param.ndim - 1
    user_r   = user_xyz.reshape([num_rx] + [1] * n_middle + [3])
    tile_m   = [1] + list(param.shape[1:]) + [1]
    user_t   = np.tile(user_r, tile_m)
    return np.concatenate([param[..., np.newaxis], user_t], axis=-1).astype(np.float32)


def _append_xyz_complex(param: np.ndarray, user_xyz: np.ndarray) -> np.ndarray:
    """Append UE (x, y, z) positions to a complex-valued RT parameter tensor.

    Parameters
    ----------
    param    : complex ndarray, shape [num_rx, ..., num_paths]
    user_xyz : float32 ndarray, shape [num_rx, 3]

    Returns
    -------
    complex64 ndarray, last dim = [param_value, x+0j, y+0j, z+0j]
    """
    num_rx   = param.shape[0]
    n_middle = param.ndim - 1
    user_r   = user_xyz.reshape([num_rx] + [1] * n_middle + [3]).astype(np.float32)
    tile_m   = [1] + list(param.shape[1:]) + [1]
    user_t   = np.tile(user_r, tile_m).astype(np.complex64)
    return np.concatenate([param[..., np.newaxis].astype(np.complex64), user_t], axis=-1)


def _zero_mask(chan: np.ndarray) -> np.ndarray:
    """Return boolean mask: True for users whose channel is all-zero (no valid paths)."""
    axes = tuple(range(1, chan.ndim))
    return (
        np.all(np.real(chan) == 0, axis=axes) &
        np.all(np.imag(chan) == 0, axis=axes)
    )


def _merge_clusters(
    cluster_files: list,
    tmp_npy: str,
    out_npz: str,
    z_value: float,
    delete_clusters: bool,
):
    """Merge per-cluster .npz files (key='filtered') into one final .npz.

    Appends the z-coordinate (receiver height) as the last element of the
    last axis, producing a final last-dim of [channel, x, y, z].
    Uses a disk-backed memory-map so RAM usage stays bounded.
    """
    cluster_files = natsorted(cluster_files)

    shapes, dtypes = [], []
    for f in cluster_files:
        with np.load(f) as npz:
            a = npz["filtered"]
            shapes.append(tuple(a.shape))
            dtypes.append(a.dtype)
        print(f"  {os.path.basename(f)}  shape={shapes[-1]}  dtype={dtypes[-1]}")

    ndim0 = len(shapes[0])
    for i, sh in enumerate(shapes):
        if len(sh) != ndim0:
            raise ValueError(
                f"ndim mismatch: {cluster_files[i]} has {len(sh)} dims vs {ndim0}"
            )
    for ax in range(1, ndim0):
        ref = shapes[0][ax]
        for i, sh in enumerate(shapes):
            if sh[ax] != ref:
                raise ValueError(
                    f"Axis-{ax} mismatch: {shapes[0]} vs {sh} "
                    f"(file {os.path.basename(cluster_files[i])})"
                )

    total_users   = sum(s[0] for s in shapes)
    out_dtype     = dtypes[0] if all(d == dtypes[0] for d in dtypes) else np.complex64
    orig_last_dim = shapes[0][-1]
    out_shape     = list(shapes[0])
    out_shape[0]  = total_users
    out_shape[-1] = orig_last_dim + 1   # append z
    out_shape     = tuple(out_shape)

    print(f"\nMerging {len(cluster_files)} clusters → shape {out_shape}  dtype {out_dtype}")
    out = np.lib.format.open_memmap(tmp_npy, mode="w+", dtype=out_dtype, shape=out_shape)

    start = 0
    for f in cluster_files:
        with np.load(f) as npz:
            arr = np.array(npz["filtered"], copy=False)
        n   = arr.shape[0]
        end = start + n
        print(f"  Copying users [{start}:{end}]  {os.path.basename(f)}")

        if arr.dtype == out_dtype:
            out[start:end, ..., :orig_last_dim] = arr
        else:
            out[start:end, ..., :orig_last_dim] = arr.astype(out_dtype)

        z_val = (np.array(z_value + 0j, dtype=out_dtype)
                 if np.issubdtype(out_dtype, np.complexfloating)
                 else np.array(z_value, dtype=out_dtype))
        out[start:end, ..., orig_last_dim] = z_val
        out.flush()
        del arr

        if delete_clusters:
            try:
                os.remove(f)
            except OSError as exc:
                print(f"  Warning: could not delete {f}: {exc}")

        start = end

    out.flush()
    del out

    with zipfile.ZipFile(out_npz, mode="w",
                         compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(tmp_npy, arcname="combined_array.npy")

    os.remove(tmp_npy)
    print(f"Saved → {out_npz}\n")


def _merge_rt_params_clusters(cluster_files: list, out_npz_params: str):
    """Merge per-cluster RT parameter arrays (AoA, AoD, ZoA, ZoD, tau, a_cir)."""
    cluster_files = natsorted(cluster_files)
    keys = ["AoA", "AoD", "ZoA", "ZoD", "tau", "a_cir"]
    buffers = {k: [] for k in keys}

    for f in cluster_files:
        with np.load(f) as npz:
            for k in keys:
                buffers[k].append(np.array(npz[k], copy=False))
        print(f"  Loaded RT params from {os.path.basename(f)}")

    # Pad num_paths axis (axis 4) to the global maximum so concatenation works
    max_paths = max(a.shape[4] for a in buffers["AoA"])
    print(f"  Max num_paths across clusters: {max_paths}")

    def pad_paths(arr, max_p):
        deficit = max_p - arr.shape[4]
        if deficit == 0:
            return arr
        pw = [(0, 0)] * arr.ndim
        pw[4] = (0, deficit)
        return np.pad(arr, pw, mode="constant", constant_values=0)

    merged = {}
    for k in keys:
        padded      = [pad_paths(a, max_paths) for a in buffers[k]]
        merged[k]   = np.concatenate(padded, axis=0)
        del buffers[k]

    np.savez_compressed(out_npz_params, **merged)
    print(f"Saved RT params → {out_npz_params}")
    for k, v in merged.items():
        print(f"  {k:8s}  shape={v.shape}  dtype={v.dtype}")


# ---------------------------------------------------------------------------
# Single-realization cluster processing
# ---------------------------------------------------------------------------

def process_cluster_single(
    scene,
    part: pd.DataFrame,
    start: int,
    cluster_idx: int,
    args,
    freqs,
    out_dir: str,
    cluster_prefix: str,
) -> str:
    """Process one cluster of UEs (single realization per UE).

    Returns the path to the saved cluster .npz file.
    """
    # --- Add receivers ---
    for rx_name in list(scene.receivers):
        scene.remove(rx_name)

    user_positions = {}
    for local_idx, row in enumerate(part.itertuples()):
        global_idx = start + local_idx
        rx_name    = f"rx_{global_idx}"
        scene.add(
            Receiver(
                name=rx_name,
                position=(float(row.x), float(row.y), float(args.rx_height)),
                orientation=(0.0, -1.0, 0.0),
            )
        )
        user_positions[rx_name] = (row.x, row.y)

    # --- Ray tracing ---
    p_solver = PathSolver()
    paths = p_solver(
        scene=scene,
        max_depth=args.max_depth,
        max_num_paths_per_src=args.max_num_paths,
        samples_per_src=args.samples_per_src,
        los=args.los,
        specular_reflection=args.specular_reflection,
        diffuse_reflection=args.diffuse_reflection,
        diffraction=args.diffraction,
        refraction=args.refraction,
        synthetic_array=args.synthetic_array,
        seed=args.base_seed,
    )

    # --- OFDM channel ---
    a_cir_raw, tau = paths.cir(normalize_delays=False, out_type="numpy")
    ofdm = paths.cfr(
        frequencies=freqs,
        normalize=False,
        normalize_delays=False,
        out_type="numpy",
    )   # shape: [num_rx, rx_ant, tx, tx_ant, steps, subcarriers]

    num_rx     = ofdm.shape[0]
    num_rx_ant = ofdm.shape[1]
    num_tx     = ofdm.shape[2]
    num_tx_ant = ofdm.shape[3]
    num_steps  = ofdm.shape[4]
    fft_size   = ofdm.shape[5]

    user_list = [user_positions[f"rx_{start + i}"] for i in range(num_rx)]
    user_xy   = np.array(user_list, dtype=np.float32)   # [num_rx, 2]
    user_xyz  = np.column_stack(
        [user_xy, np.full(num_rx, args.rx_height, dtype=np.float32)]
    )

    # Tile (x, y) to match OFDM shape and concatenate
    chan_pos  = _append_xy_complex(ofdm, user_xy)
    # shape: [num_rx, rx_ant, tx, tx_ant, steps, subcarriers, 3]

    # Filter all-zero users
    valid = ~_zero_mask(chan_pos[..., 0])
    filtered = chan_pos[valid]

    save_kwargs = {"filtered": filtered}

    # --- Optional RT parameters ---
    if args.save_rt_params:
        phi_r   = np.array(paths.phi_r)    # AoA azimuth [rad]
        phi_t   = np.array(paths.phi_t)    # AoD azimuth [rad]
        theta_r = np.array(paths.theta_r)  # ZoA zenith [rad]
        theta_t = np.array(paths.theta_t)  # ZoD zenith [rad]
        # a_cir: [rx, rx_ant, tx, tx_ant, paths, time_steps] → squeeze time_steps
        a_np    = a_cir_raw.squeeze(-1) if (a_cir_raw.ndim == 6 and a_cir_raw.shape[-1] == 1) \
                  else a_cir_raw

        save_kwargs["AoA"]   = _append_xyz_float(phi_r,                user_xyz)[valid]
        save_kwargs["AoD"]   = _append_xyz_float(phi_t,                user_xyz)[valid]
        save_kwargs["ZoA"]   = _append_xyz_float(theta_r,              user_xyz)[valid]
        save_kwargs["ZoD"]   = _append_xyz_float(theta_t,              user_xyz)[valid]
        save_kwargs["tau"]   = _append_xyz_float(tau.astype(np.float32), user_xyz)[valid]
        save_kwargs["a_cir"] = _append_xyz_complex(a_np,               user_xyz)[valid]

    out_name = os.path.join(out_dir, f"{cluster_prefix}{cluster_idx}.npz")
    np.savez_compressed(out_name, **save_kwargs)
    print(f"  [Cluster {cluster_idx}] Saved {os.path.basename(out_name)}  "
          f"shape={filtered.shape}  valid={int(valid.sum())}/{num_rx}")
    return out_name


# ---------------------------------------------------------------------------
# Multi-realization cluster processing
# ---------------------------------------------------------------------------

def process_cluster_multi(
    scene,
    part: pd.DataFrame,
    start: int,
    cluster_idx: int,
    args,
    freqs,
    out_dir: str,
    cluster_prefix: str,
) -> str:
    """Process one cluster of UEs with multiple independent realizations.

    Returns the path to the saved cluster .npz file.
    """
    for rx_name in list(scene.receivers):
        scene.remove(rx_name)

    user_positions = {}
    for local_idx, row in enumerate(part.itertuples()):
        global_idx = start + local_idx
        rx_name    = f"rx_{global_idx}"
        scene.add(
            Receiver(
                name=rx_name,
                position=(float(row.x), float(row.y), float(args.rx_height)),
                orientation=(0.0, -1.0, 0.0),
            )
        )
        user_positions[rx_name] = (row.x, row.y)

    p_solver = PathSolver()

    all_cfr = []
    for r_idx in tqdm(
        range(args.num_realizations),
        desc=f"  Realizations [cluster {cluster_idx}]",
        leave=False,
    ):
        paths = p_solver(
            scene=scene,
            max_depth=args.max_depth,
            max_num_paths_per_src=args.max_num_paths,
            samples_per_src=args.samples_per_src,
            los=args.los,
            specular_reflection=args.specular_reflection,
            diffuse_reflection=args.diffuse_reflection,
            diffraction=args.diffraction,
            refraction=args.refraction,
            synthetic_array=True,    # synthetic array required for multi-realization efficiency
            seed=args.base_seed + r_idx,
        )
        all_cfr.append(
            paths.cfr(
                frequencies=freqs,
                normalize=False,
                normalize_delays=False,
                out_type="numpy",
            )
        )

    # Stack along new realization axis (axis 4)
    # each cfr:  [rx, rx_ant, tx, tx_ant, steps, subcarriers]
    # stacked:   [rx, rx_ant, tx, tx_ant, realizations, steps, subcarriers]
    ofdm = np.stack(all_cfr, axis=4)
    del all_cfr
    gc.collect()

    num_rx     = ofdm.shape[0]
    num_rx_ant = ofdm.shape[1]
    num_tx     = ofdm.shape[2]
    num_tx_ant = ofdm.shape[3]
    num_real   = ofdm.shape[4]
    num_steps  = ofdm.shape[5]
    fft_size   = ofdm.shape[6]

    user_list = [user_positions[f"rx_{start + i}"] for i in range(num_rx)]
    user_xy   = np.array(user_list, dtype=np.float32)

    chan_pos = _append_xy_complex(ofdm, user_xy)
    # shape: [rx, rx_ant, tx, tx_ant, realizations, steps, subcarriers, 3]

    valid    = ~_zero_mask(chan_pos[..., 0])
    filtered = chan_pos[valid]

    out_name = os.path.join(out_dir, f"{cluster_prefix}{cluster_idx}.npz")
    np.savez_compressed(out_name, filtered=filtered)
    print(f"  [Cluster {cluster_idx}] Saved {os.path.basename(out_name)}  "
          f"shape={filtered.shape}  valid={int(valid.sum())}/{num_rx}")
    return out_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    # Reproducibility
    np.random.seed(args.base_seed)
    tf.random.set_seed(args.base_seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration for reproducibility
    cfg_path = os.path.join(args.output_dir, f"{args.output_prefix}_config.json")
    with open(cfg_path, "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"Configuration saved → {cfg_path}")

    print(
        f"\n{'='*60}\n"
        f"  Sionna RT Channel Generator\n"
        f"  Sionna version : {sionna.__version__}\n"
        f"  Frequency      : {args.frequency/1e9:.3f} GHz\n"
        f"  TX array       : {args.tx_rows}×{args.tx_cols} (rows×cols)\n"
        f"  RX array       : {args.rx_rows}×{args.rx_cols}\n"
        f"  Realizations   : {args.num_realizations}\n"
        f"  Save RT params : {args.save_rt_params}\n"
        f"  LoS filter     : {args.los_filter}\n"
        f"{'='*60}\n"
    )

    # --- Load scene ---
    scene = load_scene(args.scene)
    scene.frequency = args.frequency

    # --- TX configuration ---
    if args.tx_info_csv is not None:
        ant_df = pd.read_csv(args.tx_info_csv)
        tx_azimuth  = float(ant_df.loc[0, "azimuth"])
        tx_tilt     = float(ant_df.loc[0, "tilt"])
        tx_height   = float(ant_df.loc[0, "ant_height"])
        tx_x, tx_y  = args.tx_x, args.tx_y
    else:
        tx_azimuth  = args.tx_azimuth
        tx_tilt     = args.tx_tilt
        tx_height   = args.tx_height
        tx_x, tx_y  = args.tx_x, args.tx_y

    tx_orient = _tx_orientation_from_degrees(tx_azimuth, tx_tilt)

    # --- Antenna arrays ---
    scene.tx_array = PlanarArray(
        num_rows=args.tx_rows,
        num_cols=args.tx_cols,
        vertical_spacing=args.v_spacing,
        horizontal_spacing=args.h_spacing,
        pattern=args.antenna_pattern,
        polarization=args.polarization,
    )
    scene.rx_array = PlanarArray(
        num_rows=args.rx_rows,
        num_cols=args.rx_cols,
        vertical_spacing=args.v_spacing,
        horizontal_spacing=args.h_spacing,
        pattern=args.antenna_pattern,
        polarization=args.polarization,
    )

    scene.add(
        Transmitter(
            name="tx",
            position=(float(tx_x), float(tx_y), float(tx_height)),
            orientation=[float(v) for v in tx_orient],
        )
    )

    # --- Load UE positions ---
    ue_df = pd.read_csv(args.ue_csv).reset_index(drop=True)

    if args.los_filter != "all":
        if "los_ray_tracing" not in ue_df.columns:
            raise ValueError(
                f"Column 'los_ray_tracing' is required for --los_filter={args.los_filter} "
                f"but was not found in {args.ue_csv}."
            )
        before = len(ue_df)
        keep   = (ue_df["los_ray_tracing"] == True) if args.los_filter == "los" \
                 else (ue_df["los_ray_tracing"] == False)
        ue_df  = ue_df[keep].reset_index(drop=True)
        print(f"LoS filter '{args.los_filter}': {before} → {len(ue_df)} UEs retained")

    total_users  = len(ue_df)
    num_clusters = math.ceil(total_users / args.chunk_size)
    print(f"Total UEs: {total_users}  |  Clusters: {num_clusters}  "
          f"(chunk size: {args.chunk_size})\n")

    # --- OFDM subcarrier frequencies ---
    freqs = subcarrier_frequencies(args.num_subcarriers, args.subcarrier_spacing)

    # --- Determine naming ---
    multi_mode     = args.num_realizations > 1
    cluster_prefix = f"{args.output_prefix}_cluster"

    cluster_files = []

    # --- Process clusters ---
    for cluster_idx in tqdm(range(num_clusters), desc="Clusters"):
        start = cluster_idx * args.chunk_size
        end   = min(start + args.chunk_size, total_users)
        part  = ue_df.iloc[start:end].reset_index(drop=True)

        if multi_mode:
            out_name = process_cluster_multi(
                scene, part, start, cluster_idx, args, freqs,
                args.output_dir, cluster_prefix,
            )
        else:
            out_name = process_cluster_single(
                scene, part, start, cluster_idx, args, freqs,
                args.output_dir, cluster_prefix,
            )
        cluster_files.append(out_name)

    # --- Merge clusters into final dataset ---
    print("\nMerging cluster files …")

    out_npz = os.path.join(args.output_dir, f"{args.output_prefix}_channel.npz")
    tmp_npy = os.path.join(args.output_dir, "_merge_tmp.npy")

    _merge_clusters(
        cluster_files   = cluster_files,
        tmp_npy         = tmp_npy,
        out_npz         = out_npz,
        z_value         = args.rx_height,
        delete_clusters = args.delete_cluster_files,
    )

    # Merge RT parameters if they were saved
    if args.save_rt_params and not multi_mode:
        out_npz_params = os.path.join(
            args.output_dir, f"{args.output_prefix}_rt_params.npz"
        )
        print("Merging RT parameter cluster files …")
        _merge_rt_params_clusters(cluster_files, out_npz_params)

    print("\nAll done.")
    print(f"  Channel dataset : {out_npz}")
    if args.save_rt_params and not multi_mode:
        print(f"  RT parameters   : {out_npz_params}")
    print(
        "\nFinal array axes:\n"
        "  0 : users (num_rx)\n"
        "  1 : RX antennas\n"
        "  2 : TX (1)\n"
        "  3 : TX antennas\n"
        + (f"  4 : realizations ({args.num_realizations})\n"
           f"  5 : time steps\n"
           f"  6 : OFDM subcarriers ({args.num_subcarriers})\n"
           f"  7 : [channel, x_pos, y_pos, z_pos] (4)\n"
           if multi_mode else
           f"  4 : time steps\n"
           f"  5 : OFDM subcarriers ({args.num_subcarriers})\n"
           f"  6 : [channel, x_pos, y_pos, z_pos] (4)\n")
    )


if __name__ == "__main__":
    main()
