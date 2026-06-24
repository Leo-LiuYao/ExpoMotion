import torch
import torch.nn as nn
import torch.nn.functional as F 
from utils.utils import *
from einops import rearrange
import math
import torchvision
import numpy as np


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, gt, loss_map=None):
        if loss_map is not None:
            loss_map = loss_map.repeat(1, pred.shape[1], 1, 1)
            pred = pred * loss_map
            gt = gt * loss_map
            return nn.L1Loss()(pred, gt)
        else:
            return nn.L1Loss()(pred, gt)


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, patch_size=0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)
        self.ps = patch_size

    def forward(self, pred, target, mask=None):
        
        if self.ps > 0:
            B, C, H, W = pred.size()

            grid_height, grid_width = H // self.ps, W//self.ps
            pred_patch = rearrange(
                pred, "n c (gh bh) (gw bw) -> n (c gh gw) bh bw", 
                gh=grid_height, gw=grid_width, bh=self.ps, bw=self.ps) 
            
            target_patch = rearrange(
                target, "n c (gh bh) (gw bw) -> n (c gh gw) bh bw", 
                gh=grid_height, gw=grid_width, bh=self.ps, bw=self.ps) 
            
            pred_fft = torch.fft.rfft2(pred_patch, dim=(-2, -1))
            target_fft = torch.fft.rfft2(target_patch, dim=(-2, -1))

            pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        
        else:
            pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
            target_fft = torch.fft.rfft2(target, dim=(-2, -1))

            pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)
    
    
class Loss(nn.Module):
    def __init__(self, le_lambda=0.005, mu=5000):
        super(Loss, self).__init__()
        self.mu = mu
        self.le_lambda = le_lambda
        self.loss_recon = L1Loss()
        if self.le_lambda > 0:
            self.loss_le = FFTLoss()

    def forward(self, pred, gt, loss_map = None):
        loss = 0
        loss_dict = {}
        loss_recon = self.loss_recon(pred, gt, loss_map)
        loss_dict['loss_recon'] = loss_recon
        loss = loss + loss_recon
        if self.le_lambda > 0:
            loss_le = self.loss_le(pred, gt, loss_map) * self.le_lambda
            loss_dict['loss_le'] = loss_le
            loss = loss + loss_le
        return loss, loss_dict
