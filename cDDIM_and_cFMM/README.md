## Training and Inference

### Installation of conda environment to train the models
First, create the virtual environment and activate the environment. 
```ruby
conda env create --file environment.yml
```

Then, activate the environment with the command below.
```ruby
conda activate ssmimo
```

### Step 0 — Download the Datasets

The datasets used in the paper are publicly available on Hugging Face:
**[https://huggingface.co/datasets/PaulAlm/GenAI_Channel_Modeling_Datasets](https://huggingface.co/datasets/PaulAlm/GenAI_Channel_Modeling_Datasets)**. These datasets can be downloaded with the command below but note that due to the size of the files, it can take several minutes to finish. Alternatively, the individual files can be downloaded manually using the user interface.

```bash
git clone https://huggingface.co/datasets/PaulAlm/GenAI_Channel_Modeling_Datasets
```

Once downloaded, update the dataset path in the training and inference scripts to point to the directory where the dataset was downloaded.

### Step 1 — Train cDDIM

---

> **Important:** cDDIM must be trained **before** cFMM. The cFMM training relies on the same dataset indices (train/validation/test splits) that are established during cDDIM training. Running cFMM first or with a different random seed will result in mismatched data splits.

---

Train the conditional DDIM model first. This step fixes the dataset indices that must be reused in Step 2.

```bash
(ssmimo)$ python train_cDDIM_LoS.py    # LoS only scenario. Use [ssmimo] environment
(ssmimo)$ python train_cDDIM_NLoS.py   # LoS and NLoS scenario. Use [ssmimo] environment
```

### Step 2 — Train cFMM

After cDDIM training is complete, train the cFMM model. Make sure the dataset indices are consistent with those used in Step 1.

```bash
(ssmimo)$ python train_cFMM_LoS.py    # LoS only scenario. Use [ssmimo] environment
(ssmimo)$ python train_cFMM_NLoS.py   # LoS and NLoS scenario. Use [ssmimo] environment
```

### Pre-trained Models

The models used in the paper are publicly available on Hugging Face and can be downloaded to skip training entirely:

```bash
git clone https://huggingface.co/PaulAlm/GenAI_Channel_Modeling_Models
cd GenAI_Channel_Modeling_Models
unzip logs
```

Once downloaded, set the `save_dir` path in the inference scripts to point to the corresponding model directory inside `pretrained_models/` before running inference.

### Inference

Run inference after training (Steps 1–2 above) or after downloading the pre-trained models. Update the `save_dir` path to point to the corresponding model directory.

```bash
# cDDIM inference
(ssmimo)$ python infer_cDDIM_LoS.py generate    # LoS only scenario. Update the save_dir path. Use [ssmimo] environment
(ssmimo)$ python infer_cDDIM_NLoS.py generate   # LoS and NLoS scenario. Update the save_dir path. Use [ssmimo] environment

# cFMM inference
(ssmimo)$ python infer_cFMM_LoS.py generate     # LoS only scenario. Update the save_dir path. Use [ssmimo] environment
(ssmimo)$ python infer_cFMM_NLoS.py generate    # LoS and NLoS scenario. Update the save_dir path. Use [ssmimo] environment
```

These scripts will generate new images in the same directory where the logs are stored to visualize the performance of the models.

## References

This repository used code from open-source codebases such as [https://github.com/taekyunl/cDDIM](https://github.com/taekyunl/cDDIM)

Downstream tasks mentioned in the paper are inspired by:
- Channel compression - CRNet: [https://github.com/Kylin9511/CRNet](https://github.com/Kylin9511/CRNet)
- Site-specific beamforming - DLGF: [https://github.com/YuqiangHeng/DLGF](https://github.com/YuqiangHeng/DLGF)


