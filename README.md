<div align="center">

# 🌅 ExpoMotion：A Large-Scale Benchmark and A Householder Projection Network for Multi-Exposure Fusion

<!-- Official PyTorch implementation · ECCV 2026 -->

[![中文](https://img.shields.io/badge/README-中文-9e9e9e?style)](README_CN.md) [![ECCV 2026](https://img.shields.io/badge/ECCV-2026-1a73e8.svg)](https://eccv.ecva.net/) [![GitHub](https://img.shields.io/badge/GitHub-ExpoMotion-181717.svg?logo=github)](https://github.com/Leo-LiuYao/ExpoMotion)

</div>

---

<p align="center">
  <img src="assets/figure1.png" width="90%" alt="ExpoMotion teaser">
  <br>
  <em>Deviating from the conventional focus on perceptual quality in static scenes, our work introduces a large-scale MEF dataset dedicated to simultaneous detail restoration and motion artifact suppression. We integrate data from controlled and real-world motion, extreme lighting environments, and laboratory setups.</em>
</p>


## ✨ Overview

This repository contains the training and testing code for **HOP** (**H**ouseholder **O**rthogonal **P**rojection network), together with instructions for the **ExpoMotion** benchmark dataset.

## 📦 ExpoMotion Dataset

> ⚠️ The dataset is **not included** in this repository. Download `ExpoMotion.zip` from either mirror below and extract it locally.

<div align="center">

| Mirror | Link | Code |
|:------:|:----:|:----:|
| **Baidu Netdisk** | [ExpoMotion.zip](https://pan.baidu.com/s/1FV5JOPvKvc_PmDMiHIO1Ww) | `EXPO` |
| **Google Drive** | [ExpoMotion.zip](https://drive.google.com/file/d/1gY_S737bkDTdXZB8M8atWm9s5E8EUeGZ/view?usp=sharing) | — |

</div>

After extraction, set `DATA_ROOT` to the folder that contains `expomotion/` (and optionally `expomotion_resize/`):

```text
DATA_ROOT/
├── expomotion/
│   ├── training/          # 1,493 sequences, with GT
│   ├── testing_1/         # controlled motion, with GT (reference evaluation)
│   └── testing_2/         # real-world motion, without GT (no-reference evaluation)
└── expomotion_resize/     # optional resized release
```

<div align="center">

| Split | Motion type | Ground truth | Evaluation |
|:------|:-----------:|:------------:|:-----------|
| `training`  | mixed      | ✅ (`HDR.jpg`) | supervised training |
| `testing_1` | controlled | ✅            | reference-based (PSNR, SSIM, LPIPS, …) |
| `testing_2` | real-world | ❌            | no-reference inference |

</div>

Each sequence is a subfolder of JPG images. `training` and `testing_1` use `0.jpg`, `1.jpg`, `2.jpg` as inputs and `HDR.jpg` as GT. `testing_2` contains multiple input frames only (no `HDR.jpg`).

## 🚀 Training

<details open>
<summary><b>Single GPU</b></summary>

Replace `DATA_ROOT` with your local path:

```bash
python train.py \
    --dataset_name ExpoMotion \
    --model_name HOP-B \
    --logdir logdir/HOP-B \
    --train_dataset_dir DATA_ROOT/expomotion \
    --train_path training \
    --test_dataset_dir DATA_ROOT/expomotion \
    --test_path testing_1 \
    --epochs 150 \
    --phase1_epochs 150 \
    --lr 2e-4 \
    --batch_size 4 \
    --amp \
    --model_arch 1 \
    --crop_num_uniform 10 \
    --crop_num_random 20 \
    --dim 48 \
    --encoder_depth 2 \
    --num_blocks 2 4 4 \
    --heads 1 2 4
```

</details>

<details>
<summary><b>Multi-GPU (4× GPU)</b></summary>

Edit `DATA_ROOT` and hyperparameters in `train.sh`, then run:

```bash
bash train.sh
```

</details>

### ⚙️ Default training setup

<div align="center">

| Setting | Value |
|:--------|:------|
| Optimizer | Adam (`lr = 2×10⁻⁴`) |
| LR schedule | Cosine annealing over 150 epochs |
| Batch size | 4 per GPU |
| Mixed precision | `--amp` |
| Loss | L1 reconstruction |

</div>

Pretrained checkpoints (e.g., `HOP_B_ExpoMotion.pth`) are included in `ExpoMotion.zip` from the download mirrors above.

## 🧪 Testing

<details open>
<summary><b>testing_1 — controlled motion, reference evaluation (with GT)</b></summary>

```bash
python test.py \
    --model_name HOP-B \
    --dataset_name expomotion \
    --checkpoint path/to/checkpoint.pth \
    --test_dataset_dir DATA_ROOT/expomotion/testing_1 \
    --output_dir ./test_results/hop-B_testing_1 \
    --dim 48 \
    --encoder_depth 3 \
    --num_blocks 2 4 4 6 \
    --heads 1 2 4 8 \
    --ffn_expansion_factor 2.0 \
    --save_gt
```

</details>

<details>
<summary><b>testing_2 — real-world motion, no-reference inference (without GT)</b></summary>

```bash
python test.py \
    --model_name HOP-B \
    --checkpoint path/to/checkpoint.pth \
    --test_dataset_dir DATA_ROOT/expomotion/testing_2 \
    --output_dir ./test_results/hop-B_testing_2 \
    --dim 48 \
    --encoder_depth 3 \
    --num_blocks 2 4 4 6 \
    --heads 1 2 4 8
```

</details>

Predictions and PSNR/SSIM metrics (when GT is available) are saved under `{output_dir}/`.

## 🧩 Model Variants

<div align="center">

| Variant | Depth | Capacity | Use case |
|:-------:|:-----:|:--------:|:---------|
| **HOP-S** | `encoder_depth = 2` | lighter, fewer blocks | fast / lightweight |
| **HOP-B** | `encoder_depth = 3` | larger, more blocks | best quality |

</div>

> 💡 Use the same `--dim`, `--encoder_depth`, `--num_blocks`, and `--heads` at test time as in training when loading a checkpoint.

## 📁 Project Structure

```text
.
├── train.py              # Training with optional DDP
├── test.py               # Inference and PSNR/SSIM evaluation
├── train.sh / test.sh    # Example launch scripts
├── assets/
│   └── figure1.png       # Teaser figure
├── model/
│   └── hop.py            # HOP network (GPIA + HOA)
├── dataset/
│   └── datasets.py       # Train / test dataloaders
├── loss/
│   └── loss.py           # L1 (+ optional FFT) loss
├── lr_scheduler/
│   └── mylr.py           # Two-phase cosine LR scheduler
├── pytorch_ssim/
│   └── __init__.py       # SSIM implementation
└── utils/
    ├── utils.py
    └── utils_t.py
```

## 🔗 Related Links

A subset of the **ExpoMotion** dataset has been adopted by the NTIRE 2026 RAIM Challenge (Track 2) on multi-exposure image fusion in dynamic scenes:

- **Competition:** [NTIRE 2026 RAIM — Track 2 (Codabench)](https://www.codabench.org/competitions/12728/)
- **Baseline code:** [qulishen/RAIM-HDR](https://github.com/qulishen/RAIM-HDR)

## 📖 Citation

If you find ExpoMotion or HOP useful in your research, please consider citing:

```bibtex
@inproceedings{expo_motion_hop_2026,
  title     = {ExpoMotion: A Large-Scale Benchmark and A Householder Projection Network for Multi-Exposure Fusion},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```


---

<div align="center">

⭐ If this project helps you, please consider giving it a star!

</div>
