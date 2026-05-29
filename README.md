# Site-Specific MIMO Channel Generation via Diffusion and Flow Matching: Fidelity, Efficiency, and Downstream Utility

#### Link to paper (to be updated): [[TBC](TBC)]
#### Authors: Sina Beyraghi, Masoud Sadeghian, Firdous Bin Ismail, Angel Lozano, Paul Almasan, and Giovanni Geraci

Contact: Sina Beyraghi (<mohammadsina.beyraghi@telefonica.com>)

## Abstract
TBC.

## Repository Structure

```
GenAI_Channel_Modeling/
├── cDDIM_and_cFMM/          ← Generative model training & inference
├── Channel_Sionna_RT_Github/ ← Dataset generation with Sionna RT ray tracing
├── CRNet_Github/             ← Downstream task: CSI compression
├── DLGF_Github/              ← Downstream task: beam alignment
└── Effective_Rank_Github/    ← Evaluation metric: effective rank analysis
```

### [`cDDIM_and_cFMM/`](cDDIM_and_cFMM/README.md) — Generative Models

Training and inference scripts for the two generative models proposed in the paper:

- **cDDIM** — conditional Denoising Diffusion Implicit Model
- **cFMM** — conditional Flow Matching Model

Both models support LoS-only and mixed LoS+NLoS propagation scenarios. **cDDIM must be trained before cFMM** to ensure consistent train/val/test dataset splits.

### [`Channel_Sionna_RT_Github/`](Channel_Sionna_RT_Github/README.md) — Dataset Generation

Scripts to generate large-scale MIMO channel datasets from 3D radio environments using [NVIDIA Sionna RT](https://nvlabs.github.io/sionna/). Produces `.npz` files containing OFDM channel tensors and optional ray-tracing parameters (AoA, AoD, delays, CIR). These are the datasets used to train the generative models.

### [`CRNet_Github/`](CRNet_Github/README.md) — Downstream Task: CSI Compression

Extends [CRNet](https://github.com/Kylin9511/CRNet) to study how synthetically generated channels (from cDDIM, cFMM, or 3GPP stochastic models) can augment limited real channel measurements for CSI feedback compression training. Sweeps over the number of real training samples to quantify the benefit of synthetic augmentation.

### [`DLGF_Github/`](DLGF_Github/README.md) — Downstream Task: Beam Alignment

Extends the [DL-GF](https://github.com/YuqiangHeng/DLGF) grid-free beam alignment framework to support ray-tracing, 3GPP stochastic, and generative-model (cDDIM/cFMM) channel datasets. Includes combined pretraining-on-synthetic + finetuning-on-real training strategies for UPA antenna configurations.

### [`Effective_Rank_Github/`](Effective_Rank_Github%201/Effective_Rank_Github/README.md) — Evaluation: Effective Rank

Tools for computing and visualising the effective rank of UPA MIMO channels in beamspace, and comparing generative model outputs against a ground-truth reference using the Wasserstein-1 distance.

---

## How to use the code

### Cloning

Open a terminal and execute the following commands to download the github repository.

```ruby
git clone https://github.com/Telefonica-Scientific-Research/GenAI_Channel_Modeling
cd GenAI_Channel_Modeling 
```

---

## Related resources

- **Code repository:** [GenAI_Channel_Modeling](https://github.com/Telefonica-Scientific-Research/GenAI_Channel_Modeling)
- **Datasets:** [GenAI_Channel_Modeling_Datasets](https://huggingface.co/datasets/PaulAlm/GenAI_Channel_Modeling_Datasets)

---

## Citation

```bibtex
@article{beyraghi2025sitespecific,
  title   = {Site-Specific MIMO Channel Generation via Diffusion and Flow Matching:
             Fidelity, Efficiency, and Downstream Utility},
  author  = {Beyraghi, Sina and Sadeghian, Masoud and Bin Ismail, Firdous and
             Lozano, Angel and Almasan, Paul and Geraci, Giovanni},
  journal = {arXiv preprint arXiv:2510.10190},
  year    = {2025}
}
```