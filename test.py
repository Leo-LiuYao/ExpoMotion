# -*- coding:utf-8 -*-
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.datasets import Test_Dataset
from model.hop import HouseholderHDR as HOP
from utils.utils import AverageMeter
import pytorch_ssim
import cv2
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='MEF Testing: compute PSNR/SSIM and save predictions')
    # Dataset and paths
    parser.add_argument("--dataset_name", type=str, default="expomotion_resize", help="Dataset Name")
    parser.add_argument("--model_name", type=str, default="HOP", help="Model Name")
    parser.add_argument("--test_dataset_dir", type=str, required=True, help='Test dataset directory (one subfolder per sequence)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory')

    # Model hyperparameters (must match training)
    parser.add_argument('--dim', type=int, default=48, help='Base feature dimension')
    parser.add_argument('--encoder_depth', type=int, default=3, help='Encoder depth')
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[2, 4, 4, 6], help='Number of blocks per stage')
    parser.add_argument('--heads', type=int, nargs='+', default=[1, 2, 4, 8], help='Number of attention heads per stage')
    parser.add_argument('--ffn_expansion_factor', type=float, default=2.0, help='FFN expansion factor')

    # Runtime settings
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_input', action='store_true', default=False, help='Save input images')
    parser.add_argument('--save_gt', action='store_true', default=True, help='Save ground-truth images when available')

    args = parser.parse_args()
    return args


def calculate_psnr(img1, img2):
    """Calculate PSNR for torch tensors (B, C, H, W) in range [0, 1]"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(100.0, device=img1.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights (compatible with DDP `module.` prefix)."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith('module.') else k] = v

    model.load_state_dict(new_state_dict)
    epoch = checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
    print(f"Checkpoint loaded successfully! (epoch={epoch})")
    return model


@torch.no_grad()
def test_and_save(args, model, device, test_loader, output_dir):
    """Run inference, compute PSNR/SSIM when GT is available, and save predictions."""
    model.eval()

    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    if args.save_input:
        input_dir = os.path.join(output_dir, 'inputs')
        os.makedirs(input_dir, exist_ok=True)
    if args.save_gt:
        gt_dir = os.path.join(output_dir, 'ground_truth')
        os.makedirs(gt_dir, exist_ok=True)

    avg_psnr = AverageMeter()
    avg_ssim = AverageMeter()
    per_sample_lines = []

    print("\nStarting testing and saving images...")
    print(f"Predictions will be saved to: {os.path.abspath(pred_dir)}")

    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
        inputs = [img.to(device) for img in batch_data['inputs']]
        batch_inputs = torch.stack(inputs, dim=1)

        pred = model(batch_inputs)
        pred = torch.clamp(pred, 0.0, 1.0)

        has_label = 'label' in batch_data
        if has_label:
            label = torch.clamp(batch_data['label'].to(device), 0.0, 1.0)

        for i in range(pred.size(0)):
            sample_id = batch_idx * args.batch_size + i

            if has_label:
                p = pred[i:i + 1]
                g = label[i:i + 1]
                psnr_val = calculate_psnr(p, g).item()
                ssim_val = pytorch_ssim.ssim(p, g).item()
                avg_psnr.update(psnr_val, 1)
                avg_ssim.update(ssim_val, 1)
                per_sample_lines.append(
                    f"sample_{sample_id:06d}  PSNR: {psnr_val:.4f}  SSIM: {ssim_val:.4f}"
                )

            pred_np = pred[i].cpu().permute(1, 2, 0).numpy()
            pred_np = np.clip(pred_np * 255.0, 0, 255).astype(np.uint8)
            pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(pred_dir, f"sample_{sample_id:06d}_pred.jpg"), pred_bgr)

            if args.save_gt and has_label:
                gt_np = label[i].cpu().permute(1, 2, 0).numpy()
                gt_np = np.clip(gt_np * 255.0, 0, 255).astype(np.uint8)
                gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(gt_dir, f"sample_{sample_id:06d}_gt.jpg"), gt_bgr)

            if args.save_input:
                for j, input_img in enumerate(inputs):
                    in_np = input_img[i].cpu().permute(1, 2, 0).numpy()
                    in_np = np.clip(in_np * 255.0, 0, 255).astype(np.uint8)
                    in_bgr = cv2.cvtColor(in_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(input_dir, f"sample_{sample_id:06d}_input_{j}.jpg"), in_bgr)

    print("\nAll images saved.")

    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Test Metrics (PSNR / SSIM)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Checkpoint: {os.path.abspath(args.checkpoint)}\n")
        f.write(f"Dataset: {args.dataset_name} ({args.test_dataset_dir})\n")
        f.write(f"Valid samples (with GT): {avg_psnr.count}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Average PSNR: {avg_psnr.avg:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim.avg:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write("Per-sample metrics:\n")
        f.write("\n".join(per_sample_lines) + "\n")

    print("\n" + "=" * 60)
    if avg_psnr.count > 0:
        print(f"Average PSNR: {avg_psnr.avg:.4f}")
        print(f"Average SSIM: {avg_ssim.avg:.4f}")
        print(f"Valid samples (with GT): {avg_psnr.count}")
    else:
        print("No GT in the test set; predictions saved without reference metrics.")
    print(f"Metrics file: {os.path.abspath(metrics_path)}")
    print("=" * 60)


def main():
    args = get_args()

    if not args.no_cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("=" * 60)
    print("Test configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"  Output dir: {os.path.abspath(args.output_dir)}")
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Test path: {args.test_dataset_dir}")
    print(f"  dim={args.dim}  encoder_depth={args.encoder_depth}  "
          f"num_blocks={args.num_blocks}  heads={args.heads}  ffn={args.ffn_expansion_factor}")
    print("=" * 60 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading test dataset...")
    test_dataset = Test_Dataset(dataset_dir=args.test_dataset_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Test dataset loaded: {len(test_dataset)} samples\n")

    print("Initializing model...")
    model = HOP(
        dim=args.dim,
        encoder_depth=args.encoder_depth,
        num_blocks=args.num_blocks,
        heads=args.heads,
        ffn_expansion_factor=args.ffn_expansion_factor
    ).to(device)

    model = load_checkpoint(model, args.checkpoint, device)

    test_and_save(args, model, device, test_loader, args.output_dir)

    print("\nTesting finished. Results saved to:", os.path.abspath(args.output_dir))


if __name__ == '__main__':
    main()
