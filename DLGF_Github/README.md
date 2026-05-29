# DL Grid-Free (DL-GF) Beam Alignment

This repository extends the original **DL-GF** framework for grid-free beam alignment in millimeter-wave MIMO systems.

> **Original repository:** https://github.com/YuqiangHeng/DLGF  
> **Extended by:** Sina Fazel

---

## What's new in this version

- Support for multiple channel dataset types: ray-tracing (Sionna RT), stochastic 3GPP, diffusion-model generated (cDDIM), and flow-matching generated (cFMM) channels.
- Combined multi-dataset training with automatic per-sample Frobenius normalisation for scale alignment between raw and synthetic datasets.
- UPA (Uniform Planar Array) antenna configurations (4×8 BS, 2×2 UE).
- Sweep over multiple `num_probing_beam` values in a single training run.
- Separate evaluation and plotting script with caching for fast re-plotting.

---

## Repository structure

```
train.py                # Training script
evaluate_and_plot.py    # Evaluation + figure generation
DL_utils.py             # Model definitions, loss functions, training loop
beam_utils.py           # Codebook generation and SNR utilities
environment.yml         # Conda environment specification
```

---

## Environment setup

```bash
conda env create -f environment.yml
conda activate DeepMIMO_V2
```

---

## Quick start

### 1. Train

Single dataset (ray-tracing, 200 samples):
```bash
python train.py \
    --ds1 RayTracing \
    --ds1_path /path/to/channels.npz \
    --n_samples1 200 \
    --raw1 True \
    --num_probing_beam 2 4 8 12 16 20 24 28 \
    --nepoch 500 \
    --model_save_dir Saved_Models/ \
    --train_hist_save_dir Train_Hist/
```

Combined training (pretrain on synthetic, finetune on ray-tracing):
```bash
# Step 1: pretrain on generative-model channels
python train.py \
    --ds1 DiffusionModel \
    --ds1_path /path/to/cddim_channels.npz \
    --n_samples1 10000 \
    --raw1 False \
    --num_probing_beam 2 4 8 12 16 20 24 28 \
    --nepoch 500 \
    --model_save_dir Saved_Models/pretrain/

# Step 2: finetune on real ray-tracing channels
python train.py \
    --ds1 RayTracing \
    --ds1_path /path/to/rt_channels.npz \
    --n_samples1 200 \
    --raw1 True \
    --num_probing_beam 2 4 8 12 16 20 24 28 \
    --nepoch 200 \
    --model_save_dir Saved_Models/finetuned/
```

### 2. Evaluate and plot

```bash
python evaluate_and_plot.py \
    --model_dir Saved_Models/ \
    --model_tag "ds1-RayTracing_n200_rawTrue_npb{NPB}_TxRx_diagonal_FBNone_MLP_BF_loss_noise-94.0dBm_meas16.0_seed7_TX32_RX4" \
    --num_probing_beam 2 4 8 12 16 20 24 28 \
    --test_ds_path /path/to/test_channels.npz \
    --test_ds_type RayTracing \
    --curve_label "RT (200 samples)" \
    --save_cache Results/cached_results/my_run \
    --save_fig   Results/my_figure
```

Re-plot from a saved cache without re-running evaluation:
```bash
python evaluate_and_plot.py \
    --from_cache Results/cached_results/my_run.npz \
    --save_fig   Results/my_figure_v2
```

---

## Supported dataset formats

| `--ds1` type      | File format | Key / layout |
|-------------------|-------------|--------------|
| `RayTracing`      | `.npz`      | `combined_array` (N, Nr, 1, Nt, 1, n_sc, 4) |
| `Stochastic3GPP`  | `.npz`      | `ChanPos` (N, 1, Nr, 1, Nt, 1, n_sc, 4) |
| `DiffusionModel`  | `.npz`      | `channels` (N, 2, Nr, Nt) float32, beamspace |
| `FlowMatching`    | `.npz`      | `channels` (N, 2, Nr, Nt) float32, beamspace |
| `AntennaDomain`   | `.npy`      | (N, Nr, Nt) complex64, antenna domain |

---

## Citation

If you use this code, please cite the original DL-GF paper:

```bibtex
@article{heng2023grid,
  title   = {Grid-Free MIMO Beam Alignment through Deep Learning},
  author  = {Heng, Yuqiang and Andrews, Jeffrey G.},
  journal = {IEEE Transactions on Wireless Communications},
  year    = {2023}
}
```
