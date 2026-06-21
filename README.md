# SGGCMNet: Spectral Group-Wise Gated CNN–Mamba Network

<div align="center">

[Yan Zhang](https://www.mdpi.com/2072-4292/18/11/1814), [Xianghai Cao](https://www.mdpi.com/2072-4292/18/11/1814)

<a href='https://www.mdpi.com/2072-4292/18/11/1814'><img src='https://img.shields.io/badge/Remote%20Sensing-Paper-blue'></a>

</div>

## Introduction

SGGCMNet is a hyperspectral image (HSI) classification network that combines CNN and Mamba (state-space model) branches. Key contributions:

- **CMSB (CNN–Mamba Spectral Group-wise Gating Block)**: Partitions spectral channels into sub-groups and learns per-group gating weights to adaptively balance local CNN features and long-range Mamba context, improving the discrimination of similar land-cover classes.
- **Progressive Deep Supervision** with uncertainty-based dynamic loss weighting, alleviating vanishing gradients in shallow layers.
- **TCMD (Temperature-Regulated Cross-Stage Mutual Distillation)**: Enables bidirectional knowledge transfer across network stages via temperature-softened symmetric KL divergence.

On three benchmark HSI datasets, SGGCMNet achieves state-of-the-art accuracy and remains the best across training ratios from 1% to 20%.

## Requirements

- Python ≥ 3.10
- CUDA ≥ 11.8 (verified on CUDA 12.6)
- NVIDIA GPU

```bash
# Install PyTorch with CUDA 12
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** `causal_conv1d` and `mamba_ssm` require pre-built wheels matching your torch/CUDA version. Download from
> [causal-conv1d releases](https://github.com/Dao-AILab/causal-conv1d/releases) and
> [mamba releases](https://github.com/state-spaces/mamba/releases).

## Training

Place the dataset `.mat` files in a `dataset/` directory, then run:

```bash
# Indian Pines (3% training samples)
python train.py --dataset IndianPines

# Pavia University (0.5% training samples)
python train.py --dataset PaviaUniversity

# Houston 2013 (2% training samples)
python train.py --dataset Houston
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | IndianPines | IndianPines, PaviaUniversity, Houston |
| `--epochs` | 100 | Number of training epochs |
| `--runs` | 10 | Number of independent runs |
| `--lr` | 1e-3 | Learning rate |
| `--batch_size` | 64 | Batch size |
| `--patch_size` | 11 | Spatial patch size |
| `--pca_bands` | 30 | PCA components |
| `--hidden_dim` | 64 | Hidden feature dimension |
| `--seed` | 42 | Random seed |

Results (OA, AA, Kappa, per-class accuracy, confusion matrix) are saved to `results/`.

## Main Results

| Dataset | OA (%) | AA (%) | Kappa × 100 |
|---------|--------|--------|-------------|
| Indian Pines (3% labels) | 94.96 ± 0.80 | 93.80 ± 1.21 | 94.25 ± 0.92 |
| Pavia University (0.5% labels) | 96.34 ± 0.57 | 93.51 ± 1.22 | 95.14 ± 0.76 |
| Houston 2013 (2% labels) | 94.55 ± 1.02 | 94.70 ± 0.87 | 94.11 ± 1.11 |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@Article{rs18111814,
AUTHOR = {Zhang, Yan and Cao, Xianghai},
TITLE = {A Spectral Group-Wise Gated CNN--Mamba Network with Cross-Stage Mutual Distillation for Hyperspectral Image Classification},
JOURNAL = {Remote Sensing},
VOLUME = {18},
YEAR = {2026},
NUMBER = {11},
ARTICLE-NUMBER = {1814},
URL = {https://www.mdpi.com/2072-4292/18/11/1814},
ISSN = {2072-4292},
DOI = {10.3390/rs18111814}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
