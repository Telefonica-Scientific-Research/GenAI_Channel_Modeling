# CRNet — CSI Compression with Synthetic Data Augmentation

Adapted from the original CRNet repository:
> **CRNet: An Efficient Multi-task Learning Architecture for Large-Scale MIMO CSI Feedback**
> W. Liu *et al.*, IEEE TCCN, 2022.
> Source: https://github.com/Kylin9511/CRNet

This repository extends CRNet to study how **synthetically generated channels** (from diffusion-based or flow-based generative models) can augment limited real channel measurements for CSI compression training.
Both **LoS-only** and **LoS + NLoS** propagation conditions are supported.

---

## What this code does

Given a small pool of **real (ground-truth) channel measurements** and a larger set of **synthetic channels** from a generative model (e.g. cDDIM, Flow Matching, or 3GPP stochastic), we:

1. Fix the total training budget at `N_total = 10 000` samples.
2. Sweep `N_real ∈ {200, 500, 1k, 2k, 5k, 10k}` real samples, filling the rest with synthetic data.
3. Train CRNet at each point (averaged over 3 random seeds).
4. Compare NMSE curves across methods to evaluate how well synthetic augmentation compensates for scarce real data.

The **benchmark** is a CRNet trained on 10 000 real samples — the practical upper bound.

---

## Repository structure

```
CRNet_Github/
├── train.py          # Training: one run per synthetic method
├── plot_results.py   # Plotting: reads cached JSONs, no retraining needed
├── environment.yml   # Conda environment
└── README.md
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate Ch_Comp
```

> **GPU note:** the environment was exported with `torch==1.9.1+cu111`.
> For a different CUDA version install PyTorch separately:
> ```bash
> conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
> ```

---

## Supported data formats

| `--real_format` / `--synth_format` | Description |
|------------------------------------|-------------|
| `sionna_npz`    | Sionna RT output — key `combined_array`, shape `(N, Nr, 1, Nt, 1, Nsc, 4)` |
| `3gpp_npz`      | 3GPP output — key `ChanPos`, shape `(N, 1, Nr, 1, Nt, 1, Nsc, 4)` |
| `beamspace_npz` | cDDIM / FMM output — key `channels`, shape `(N, 2, Nr, Nt)` float32 in beamspace |
| `npy_complex`   | `.npy` of shape `(N, Nr, Nt)` complex64, antenna domain |
| `npy_stacked`   | `.npy` of shape `(N, 2, Nr, Nt)` float32, real+imag on axis 1, antenna domain |

---

## Usage

### Step 1 — Train one method

**LoS-only, augment with cDDIM synthetic data:**
```bash
python train.py \
    --condition   los \
    --real_los    /path/to/sionna_los.npz  --real_format  sionna_npz \
    --synth_los   /path/to/cddim_los.npz   --synth_format beamspace_npz \
    --method_name cDDIM \
    --logs_dir Logs_LoS --plots_dir Plots_LoS
```

**LoS+NLoS, augment with Flow Matching (FMM) synthetic data:**
```bash
python train.py \
    --condition   losnlos \
    --real_los    /path/to/sionna_los.npz   --real_format  sionna_npz \
    --real_nlos   /path/to/sionna_nlos.npz \
    --synth_los   /path/to/fmm_los.npz      --synth_format beamspace_npz \
    --synth_nlos  /path/to/fmm_nlos.npz \
    --method_name "Flow Matching" \
    --logs_dir Logs --plots_dir Plots
```

**Real-only (Sionna RT) baseline — omit `--synth_*`:**
```bash
python train.py \
    --condition   los \
    --real_los    /path/to/sionna_los.npz  --real_format sionna_npz \
    --method_name "Sionna RT" \
    --logs_dir Logs_LoS --plots_dir Plots_LoS
```

**3GPP stochastic baseline:**
```bash
python train.py \
    --condition    los \
    --real_los     /path/to/sionna_los.npz   --real_format  sionna_npz \
    --synth_los    /path/to/3gpp_los.npz     --synth_format 3gpp_npz \
    --method_name  "3GPP Stochastic" \
    --logs_dir Logs_LoS --plots_dir Plots_LoS
```

Run `python train.py --help` for all options.
Results are cached to JSON — rerunning skips already-trained scenarios automatically.

---

### Step 2 — Plot results

After running one or more methods, generate the NMSE vs. N_real figure:

```bash
# LoS-only
python plot_results.py --condition los --logs_dir Logs_LoS --plots_dir Plots_LoS

# LoS+NLoS
python plot_results.py --condition losnlos --logs_dir Logs --plots_dir Plots
```

The script **auto-discovers all methods** from the log files — no need to list them manually. You can also run it after each `train.py` call to see incremental results.

---

## Key CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--condition` | `los` | `los` or `losnlos` |
| `--reduction` | `4` | Compression ratio γ |
| `--epochs` | `500` | Training epochs |
| `--n_real_sizes` | `200 500 1000 2000 5000 10000` | N_real sweep points |
| `--total_train` | `10000` | Fixed total training size (augmented methods) |
| `--nrx_x / nrx_y` | `2 / 2` | UE UPA dimensions (Nr = nrx_x × nrx_y) |
| `--ntx_x / ntx_y` | `4 / 8` | BS UPA dimensions (Nt = ntx_x × ntx_y) |
| `--no_skip` | — | Re-train all scenarios even if cached |

---

## NMSE definition

NMSE is computed in the beamspace domain using the unitary 2-D UPA DFT:

$$\text{NMSE} = \mathbb{E}\!\left[\frac{\|\mathbf{H}_v - \hat{\mathbf{H}}_v\|_F^2}{\|\mathbf{H}_v\|_F^2}\right]$$

where $\mathbf{H}_v = \mathbf{A}_r^H \mathbf{H}\,\mathbf{A}_t$ is the beamspace channel.
Because the UPA DFT is unitary, this equals the antenna-domain NMSE.

---

## Citation

If you use this code, please cite the original CRNet paper:
```bibtex
@article{liu2022crnet,
  title   = {CRNet: An Efficient Multi-task Learning Architecture for Large-Scale MIMO CSI Feedback},
  author  = {Liu, Weihao and others},
  journal = {IEEE Transactions on Cognitive Communications and Networking},
  year    = {2022}
}
```
