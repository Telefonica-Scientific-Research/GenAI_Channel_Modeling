# Sionna RT Channel Dataset Generator

Generate large-scale MIMO channel datasets from 3D radio environments using [NVIDIA Sionna RT](https://nvlabs.github.io/sionna/) — an open-source, GPU-accelerated ray tracing engine for wireless channel simulation.

---

## What this repository contains

| File | Purpose |
|---|---|
| `channel_generation.py` | **Dataset generation** — runs Sionna RT ray tracing and saves OFDM channel tensors + optional RT parameters |
| `plot_channels.py` | **Visualization** — loads generated datasets and produces coverage maps, CDFs, angular / delay spread plots |
| `environment.yml` | Conda environment with all required dependencies |

---

## Environment setup

```bash
conda env create -f environment.yml
conda activate sionna12
```

> The environment was exported from a working setup with **Sionna 1.x**, TensorFlow 2.x, and CUDA.  
> Requires a CUDA-capable GPU.

---

## Quick start

### 1 — Generate channel data

```bash
python channel_generation.py \
    --scene       scene.xml \
    --ue_csv      ue_positions.csv \
    --frequency   28e9 \
    --tx_rows 1   --tx_cols 32 \
    --rx_rows 1   --rx_cols 4  \
    --los_filter  all \
    --output_prefix  channels_28GHz
```

The script processes UEs in memory-efficient chunks and produces a single merged `.npz` file.

### 2 — Save full RT parameters (AoA, AoD, ZoA, ZoD, delay, CIR)

```bash
python channel_generation.py \
    --scene scene.xml --ue_csv ue_positions.csv \
    --save_rt_params
```

### 3 — Multi-realization mode (independent snapshots per UE)

```bash
python channel_generation.py \
    --scene scene.xml --ue_csv ue_positions.csv \
    --num_realizations 20
```

### 4 — Visualize the dataset

```bash
# Basic coverage map and power CDF
python plot_channels.py --channel_file channels_28GHz_channel.npz

# Include RT parameter plots (delay spread, angular histograms)
python plot_channels.py \
    --channel_file   channels_28GHz_channel.npz \
    --rt_params_file channels_28GHz_rt_params.npz \
    --output_dir     plots/
```

---

## Key parameters

| Flag | Default | Description |
|---|---|---|
| `--frequency` | `28e9` | Carrier frequency [Hz]. Any value works: 3.5e9, 6e9, 28e9, 60e9, … |
| `--tx_rows / --tx_cols` | `1 / 32` | TX antenna array dimensions |
| `--rx_rows / --rx_cols` | `1 / 4` | RX antenna array dimensions |
| `--los_filter` | `all` | `los`, `nlos`, or `all` — filters UEs by LoS status |
| `--num_subcarriers` | `1` | OFDM subcarriers |
| `--subcarrier_spacing` | `60e3` | Subcarrier spacing [Hz] |
| `--max_depth` | `3` | Ray interaction depth (reflections, diffractions) |
| `--num_realizations` | `1` | Independent Monte-Carlo realizations per UE |
| `--save_rt_params` | off | Export AoA, AoD, ZoA, ZoD, tau, CIR coefficients |
| `--chunk_size` | `200` | UEs per processing chunk (memory control) |

Run `python channel_generation.py --help` or `python plot_channels.py --help` for the full parameter list.

---

## Input file format

**Scene** (`--scene`): Mitsuba XML scene file with material properties.

**UE positions** (`--ue_csv`): CSV with at minimum columns `x`, `y`.  
Optional column `los_ray_tracing` (boolean) is required when `--los_filter los` or `--los_filter nlos` is used.

**TX info** (`--tx_info_csv`, optional): CSV whose first row contains `azimuth` [deg], `tilt` [deg], `ant_height` [m].  
If omitted, use `--tx_azimuth`, `--tx_tilt`, `--tx_height` directly.

---

## Output array layout

**OFDM channel** (`*_channel.npz`, key `combined_array`):

Single realization:
```
[num_users, rx_ant, num_tx, tx_ant, time_steps, subcarriers, 4]
 last dim: [complex_channel, x_pos, y_pos, z_pos]
```

Multi-realization:
```
[num_users, rx_ant, num_tx, tx_ant, realizations, time_steps, subcarriers, 4]
```

**RT parameters** (`*_rt_params.npz`):
```
Keys: AoA, AoD, ZoA, ZoD, tau, a_cir
Shape (each): [num_users, rx_ant, num_tx, tx_ant, num_paths, 4]
              last dim: [parameter_value, x_pos, y_pos, z_pos]
```

---

## Acknowledgements

Channel data generated using **NVIDIA Sionna RT** — an open-source differentiable ray tracing library for wireless communications research.

- Paper: [Sionna: An Open-Source Library for Next-Generation Physical Layer Research](https://arxiv.org/abs/2203.11854)
- Docs: [https://nvlabs.github.io/sionna/](https://nvlabs.github.io/sionna/)
- License: Apache 2.0
