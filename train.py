# -*- coding:utf-8 -*-
import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dataset.datasets import Train_Dataset, Test_Dataset
from model.hop import HouseholderHDR as HOP
from loss.loss import Loss
from utils.utils import get_logger, AverageMeter
import pytorch_ssim
import pyiqa
import cv2
import numpy as np
import swanlab
from tqdm import tqdm
from lr_scheduler.mylr import MyLR


def get_args():
    parser = argparse.ArgumentParser(description='MEF Training Settings')
    # Dataset and Path settings
    parser.add_argument("--dataset_name", type=str, default="ExpoMotion", help="Dataset Name")
    parser.add_argument("--model_name", type=str, default="HOP", help="Model Name")
    parser.add_argument('--logdir', type=str, default='logdir/HOP', help='Log directory')
    parser.add_argument("--train_dataset_dir", type=str, default='', help='Train dataset root')
    parser.add_argument('--train_path', type=str, default='', help='Train dataset subdir')
    parser.add_argument("--test_dataset_dir", type=str, default='', help='Test dataset root')
    parser.add_argument('--test_path', type=str, default='', help='Test dataset subdir')
    
    # Training settings
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--test_num_workers', type=int, default=1, help='Number of test data loading workers')
    parser.add_argument('--start_epoch', type=int, default=1, help='Start epoch')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--phase1_epochs', type=int, default=50, help='Epochs for phase 1 of LR scheduler')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Test batch size')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint')
    parser.add_argument('--seed', type=int, default=443, help='Random seed')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lr_decay', action='store_true', default=True, help='Enable learning rate decay')
    
    # GPU settings
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--use_swanlab', action='store_true', default=False, help='Enable Swanlab logging')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--amp', action='store_true', default=False, help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--model_arch', type=int, default=2, help='Model Architecture ID: 1 for SAFNet, 2 for HDRDualUGDFN, 3 for Restormer, 4 for oppo, 5 for HierarchicalMoEFusion, 6 for svdhdr')
    parser.add_argument('--crop_num_uniform', type=int, default=20, help='crop_num_uniform')
    parser.add_argument('--crop_num_random', type=int, default=10, help='crop_num_random')
    parser.add_argument('--dim', type=int, default=32, help='Base feature dimension')
    parser.add_argument('--encoder_depth', type=int, default=3, help='Encoder depth')
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[2], help='Encoder blocks per stage (1 value or per-stage list)')
    parser.add_argument('--heads', type=int, nargs='+', default=[2], help='Encoder blocks per stage (1 value or per-stage list)')
    
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def calculate_psnr(img1, img2):
    """Calculate PSNR for torch tensors (B, C, H, W) in range [0, 1]"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def train(args, model, device, train_loader, optimizer, epoch, criterion, logger, scaler):
    model.train()
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)
    
    # Only show progress bar on rank 0
    iterator = train_loader
    if args.local_rank <= 0:
        iterator = tqdm(train_loader, ncols=100, desc=f"Train Epoch {epoch}")
        
    for batch_idx, batch_data in enumerate(iterator):
        inputs = [img.to(device) for img in batch_data['inputs']]
        label = batch_data['label'].to(device)
        
        batch_inputs = torch.stack(inputs, dim=1)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(batch_inputs)
            loss, loss_dict = criterion(pred, label)
            loss = loss / args.grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % args.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        if batch_idx % args.log_interval == 0 and args.local_rank <= 0:
            log_loss = loss.item() * args.grad_accum_steps
            if args.use_swanlab:
                swanlab.log({
                    'train/loss': log_loss,
                    'train/loss_recon': loss_dict['loss_recon'].item(),
                    'train/loss_le': loss_dict['loss_le'].item() if 'loss_le' in loss_dict else 0,
                    'train/epoch': epoch
                })
            logger.info(f"Epoch: {epoch} [{batch_idx}/{len(train_loader)}] Loss: {log_loss:.6f} Loss_Recon: {loss_dict['loss_recon'].item():.6f} Loss_LE: {loss_dict['loss_le'].item() if 'loss_le' in loss_dict else 0:.6f}")

# test() receives pre-initialized IQA metrics to avoid redundant downloads under DDP
def test(args, model, device, epoch, test_loader, ckpt_dir, logger, lpips_metric, niqe_metric):
    model.eval()
    avg_psnr = AverageMeter()
    avg_ssim = AverageMeter()
    avg_lpips = AverageMeter()
    avg_niqe = AverageMeter()
    
    # Directory for visualizations
    vis_dir = os.path.join(ckpt_dir, 'visualizations', f'epoch_{epoch:03d}')
    if args.local_rank <= 0:
        os.makedirs(vis_dir, exist_ok=True)
    if args.distributed:
        dist.barrier()
    if args.local_rank > 0:
        os.makedirs(vis_dir, exist_ok=True)
    
    iterator = test_loader
    if args.local_rank <= 0:
        iterator = tqdm(test_loader, ncols=100, desc=f"Test Epoch {epoch}")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(iterator):
            inputs = [img.to(device) for img in batch_data['inputs']]
            batch_inputs = torch.stack(inputs, dim=1)
            
            pred = model(batch_inputs)
            pred = torch.clamp(pred, 0.0, 1.0)
            
            if 'label' in batch_data:
                label = batch_data['label'].to(device)
                label = torch.clamp(label, 0.0, 1.0)
                psnr = calculate_psnr(pred, label)
                ssim_val = pytorch_ssim.ssim(pred, label)
                avg_psnr.update(psnr.item(), pred.size(0))
                avg_ssim.update(ssim_val.item(), pred.size(0))
                lpips_val = lpips_metric(pred, label)
                avg_lpips.update(lpips_val.mean().item(), pred.size(0))
            
            niqe_val = niqe_metric(pred)
            avg_niqe.update(niqe_val.mean().item(), pred.size(0))
            
            if args.local_rank <= 0 or args.distributed:
                for i in range(pred.size(0)):
                    pred_np = pred[i].cpu().permute(1, 2, 0).numpy()
                    pred_np = np.clip(pred_np * 255.0, 0, 255).astype(np.uint8)
                    pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

                    prefix = f"rank{args.rank}_" if args.distributed else ""
                    stem = f"{prefix}sample_{batch_idx:06d}_{i:02d}"
                    cv2.imwrite(os.path.join(vis_dir, f"{stem}_pred.jpg"), pred_bgr)

                    if 'label' in batch_data:
                        gt_np = label[i].cpu().permute(1, 2, 0).numpy()
                        gt_np = np.clip(gt_np * 255.0, 0, 255).astype(np.uint8)
                        gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(vis_dir, f"{stem}_gt.jpg"), gt_bgr)

    if args.distributed:
        metrics = torch.tensor([avg_psnr.avg, avg_ssim.avg, avg_lpips.avg, avg_niqe.avg], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= dist.get_world_size()
        avg_psnr_val = metrics[0].item()
        avg_ssim_val = metrics[1].item()
        avg_lpips_val = metrics[2].item()
        avg_niqe_val = metrics[3].item()
    else:
        avg_psnr_val = avg_psnr.avg
        avg_ssim_val = avg_ssim.avg
        avg_lpips_val = avg_lpips.avg
        avg_niqe_val = avg_niqe.avg

    if args.local_rank <= 0:
        print(f"\nEpoch {epoch} Test PSNR: {avg_psnr_val:.4f} SSIM: {avg_ssim_val:.4f} LPIPS: {avg_lpips_val:.4f} NIQE: {avg_niqe_val:.4f}")
        
        if args.use_swanlab:
            swanlab.log({
                'test/psnr': avg_psnr_val,
                'test/ssim': avg_ssim_val,
                'test/lpips': avg_lpips_val,
                'test/niqe': avg_niqe_val,
                'test/epoch': epoch
            })
        logger.info(f"Epoch {epoch} Test PSNR: {avg_psnr_val:.4f} SSIM: {avg_ssim_val:.4f} LPIPS: {avg_lpips_val:.4f} NIQE: {avg_niqe_val:.4f}")
    
    return avg_psnr_val

def main():
    args = get_args()
    
    # Distributed Setup
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif args.local_rank != -1:
        args.distributed = True
        args.rank = args.local_rank 
        args.world_size = 1 
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = -1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device('cuda', args.local_rank)
        print(f"Initialized Distributed Training: Rank {args.rank}, Local Rank {args.local_rank}, World Size {args.world_size}")
    else:
        if not args.no_cuda and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print(f"Using device: {device}")

    if args.local_rank <= 0 and args.use_swanlab:
        swanlab.init(
            project="MEF_NEW_ablation",
            experiment_name=f"{args.model_name}_{args.dataset_name}",
            config=vars(args)
        )
    
    set_random_seed(args.seed + args.rank)
    
    if args.local_rank <= 0:
        os.makedirs(args.logdir, exist_ok=True)
        ckpt_dir = os.path.join(args.logdir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        logger_train = get_logger('train', args.logdir)
        logger_test = get_logger('test', args.logdir)
    else:
        ckpt_dir = "" 
        logger_train = None
        logger_test = None
        
    if args.distributed:
        dist.barrier()
        if args.local_rank > 0:
             class DummyLogger:
                 def info(self, msg): pass
             logger_train = DummyLogger()
             logger_test = DummyLogger()

    # Datasets
    if args.local_rank <= 0:
        print("Loading datasets...")
    
    train_dataset = Train_Dataset(dataset_dir=os.path.join(args.train_dataset_dir, args.train_path),
                                    crop_num_uniform=args.crop_num_uniform, crop_num_random=args.crop_num_random, crop_size=(256, 256), rotate_range=None)
    
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False 
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_dataset = Test_Dataset(dataset_dir=os.path.join(args.test_dataset_dir, args.test_path))
    
    if args.distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.test_num_workers,
        pin_memory=True
    )
    
    # Model
    if args.local_rank <= 0:
        print("Initializing model...")
    upscale = 4
    window_size = 8
    height = (256 // upscale // window_size + 1) * window_size
    width = (256 // upscale // window_size + 1) * window_size
    model_dict = {
        1: HOP(
            dim=args.dim, 
            encoder_depth=args.encoder_depth,
            num_blocks=args.num_blocks,
            heads=args.heads,
            ffn_expansion_factor=2.0
        ),
    }
    model = model_dict[args.model_arch].to(device)
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = MyLR(optimizer, T_max=args.epochs, phase1_epoch=args.phase1_epochs, eta_min=1e-6) if args.lr_decay else None
    
    criterion = Loss(le_lambda=0.0).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Initialize IQA metrics once before the training loop (DDP-safe download)
    # Rank 0 downloads weights first; other ranks load from local cache after barrier
    lpips_metric = None
    niqe_metric = None
    
    if args.local_rank <= 0:
        print("Initializing IQA metrics (Downloading if needed)...")
        lpips_metric = pyiqa.create_metric('lpips', as_loss=False).to(device)
        niqe_metric = pyiqa.create_metric('niqe', as_loss=False).to(device)

    if args.distributed:
        dist.barrier()

    if args.local_rank > 0:
        lpips_metric = pyiqa.create_metric('lpips', as_loss=False).to(device)
        niqe_metric = pyiqa.create_metric('niqe', as_loss=False).to(device)
    
    start_epoch = args.start_epoch
    if args.resume and os.path.isfile(args.resume):
        if args.local_rank <= 0:
            print(f"Loading checkpoint: {args.resume}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank} if args.distributed else None
        checkpoint = torch.load(args.resume, map_location=map_location)
        
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.lr_decay and 'lr_scheduler' in checkpoint and lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training Loop
    best_psnr = 0.0
    
    psnr = test(args, model, device, 0, test_loader, ckpt_dir, logger_test, lpips_metric, niqe_metric)
    
    for epoch in range(start_epoch, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion, logger_train, scaler)
        
        if lr_scheduler:
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if args.local_rank <= 0 and args.use_swanlab:
                swanlab.log({'train/lr': current_lr}, step=epoch)
        
        psnr = test(args, model, device, epoch, test_loader, ckpt_dir, logger_test, lpips_metric, niqe_metric)
        
        if args.local_rank <= 0:
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                    'scaler': scaler.state_dict(),
                    'best_psnr': best_psnr
                }, os.path.join(ckpt_dir, 'best_model.pth'))
                print(f"New best PSNR: {best_psnr:.4f} (Epoch {epoch})")
                
            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                'scaler': scaler.state_dict()
            }, os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))

if __name__ == '__main__':
    main()
