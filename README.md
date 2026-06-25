# ExpoMotion: A Large-Scale Benchmark and Householder Projection Network for Multi-Exposure Fusion

Official PyTorch implementation of **ExpoMotion: A Large-Scale Benchmark and Householder Projection Network for Multi-Exposure Fusion** (ECCV 2026).

This repository contains the **HOP** (Householder Orthogonal Projection network) training and testing code. The **ExpoMotion** benchmark dataset is hosted separately (see below).

HOP tackles dynamic multi-exposure fusion (MEF) deghosting via:

- **GPIA** (Global Priors Illumination Alignment): harmonizes exposure discrepancies using global illumination statistics.
- **HOA** (Householder Orthogonal Attention): models ghosting as orthogonal perturbations and projects them out of the feature manifold.

## ExpoMotion Dataset

The ExpoMotion dataset is **not included in this repository**. Download `ExpoMotion.zip` from either link below and extract it locally:

> **Baidu Netdisk:** [ExpoMotion.zip](https://pan.baidu.com/s/1FV5JOPvKvc_PmDMiHIO1Ww) (extraction code: `EXPO`)  
> **Google Drive:** [ExpoMotion.zip](https://drive.google.com/file/d/1gY_S737bkDTdXZB8M8atWm9s5E8EUeGZ/view?usp=sharing)

After extraction, set `DATA_ROOT` to the folder that contains `expomotion/` (and optionally `expomotion_resize/`):

```
DATA_ROOT/
├── expomotion/
│   ├── training/     # 1,493 sequences, with GT
│   ├── testing_1/    # controlled motion, with GT (reference evaluation)
│   └── testing_2/    # real-world motion, without GT (no-reference evaluation)
└── expomotion_resize/   # optional resized release
```

| Split | Motion type | Ground truth | Evaluation |
|-------|-------------|--------------|------------|
| `training` | mixed | Yes (`HDR.jpg`) | supervised training |
| `testing_1` | controlled | Yes | reference-based (PSNR, SSIM, LPIPS, …) |
| `testing_2` | real-world | No | no-reference (NIQE, MUSIQ, DeQA, …) |

Each sequence is a subfolder of JPG images. Training and `testing_1` use `0.jpg`, `1.jpg`, `2.jpg` as inputs and `HDR.jpg` as GT. `testing_2` contains multiple input frames only (no `HDR.jpg`).

## Training

Single-GPU example (replace `DATA_ROOT` with your local path):

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

Multi-GPU (4× GPU): edit `DATA_ROOT` and hyperparameters in `train.sh`, then run:

```bash
bash train.sh
```

| Setting | Value |
|---------|-------|
| Optimizer | Adam (`lr=2×10⁻⁴`) |
| LR schedule | Cosine annealing over 150 epochs |
| Batch size | 4 per GPU |
| Mixed precision | `--amp` |
| Loss | L1 reconstruction |

Checkpoints are saved under `{logdir}/ckpt/` (`best_model.pth`, `epoch_*.pth`).

## Testing

**testing_1** — controlled motion, reference evaluation (with GT):

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

**testing_2** — real-world motion, no-reference inference (without GT):

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

Predictions and metrics (when GT is available) are saved under `{output_dir}/`.

## Project Structure

```
├── train.py              # Training with optional DDP
├── test.py               # Inference and PSNR/SSIM evaluation
├── train.sh / test.sh    # Example launch scripts
├── model/
│   └── hop.py            # HOP network (GPIA + HOA)
├── dataset/
│   └── datasets.py       # Train / test dataloaders
├── loss/
│   └── loss.py           # L1 (+ optional FFT) loss
├── lr_scheduler/
│   └── mylr.py           # Two-phase cosine LR scheduler
├── metric/               # Evaluation utilities (DeQA, TMQI, etc.)
└── utils/
    └── utils.py
```

## Model Variants

- **HOP-S**: lighter (`encoder_depth=2`, fewer blocks)
- **HOP-B**: larger (`encoder_depth=3`, more blocks)

Use the same `--dim`, `--encoder_depth`, `--num_blocks`, and `--heads` as in training when loading a checkpoint.

## Citation

```bibtex
@inproceedings{expo_motion_hop_2026,
  title={ExpoMotion: A Large-Scale Benchmark and Householder Projection Network for Multi-Exposure Fusion},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

Third-party code under `metric/DeQA/` follows its own license (see `metric/DeQA/LICENSE`).
