"""
Perturbation operators from the paper.
Mask m: m=1 means keep original, m=0 means perturb (delete).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gaussian_kernel_2d(sigma, kernel_size=None):
    """Create 2D Gaussian kernel for blur."""
    if kernel_size is None:
        kernel_size = max(3, int(2 * np.ceil(3 * sigma) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)  # (K,K)
    return kernel


class GaussianBlur(nn.Module):
    """Differentiable Gaussian blur (sigma fixed)."""

    def __init__(self, sigma=10, kernel_size=None):
        super().__init__()
        self.sigma = sigma
        k = gaussian_kernel_2d(sigma, kernel_size)
        self.register_buffer("kernel", k.unsqueeze(0).unsqueeze(0))  # (1,1,K,K)

    def forward(self, x):
        # x: (B,C,H,W)
        k = self.kernel.expand(x.size(1), 1, -1, -1)  # (C,1,K,K)
        padding = self.kernel.size(-1) // 2
        return F.conv2d(x, k, padding=padding, groups=x.size(1))


def apply_blur_perturbation(img, mask, sigma=10, blur_module=None):
    """
    Perturbation: perturbed = mask * img + (1 - mask) * blurred_img.
    mask=1 keeps original, mask=0 replaces with blur.
    Paper uses sigma=10 for the blur kernel.
    """
    if blur_module is None:
        blur_module = GaussianBlur(sigma=sigma).to(img.device)
    blurred = blur_module(img)
    return mask * img + (1 - mask) * blurred


def apply_constant_perturbation(img, mask, constant=None):
    """
    Perturbation: perturbed = mask * img + (1 - mask) * mu0.
    constant mu0: in normalized space, use 0 (mean); or pass custom (1,3,1,1).
    """
    if constant is None:
        constant = torch.zeros(1, 3, 1, 1, device=img.device, dtype=img.dtype)
    return mask * img + (1 - mask) * constant


def apply_noise_perturbation(img, mask, sigma=0.2):
    """
    Perturbation: perturbed = mask * img + (1 - mask) * noise.
    noise is i.i.d. Gaussian.
    """
    noise = torch.randn_like(img, device=img.device) * sigma
    return mask * img + (1 - mask) * noise


def upsample_mask(mask, size, smooth_sigma=5):
    """
    Upsample low-res mask to image size and optionally smooth.
    mask: (B, 1, h, w)
    size: (H, W)
    """
    up = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    if smooth_sigma > 0:
        k = gaussian_kernel_2d(smooth_sigma)
        k = k.unsqueeze(0).unsqueeze(0).to(mask.device)
        k = k.expand(1, 1, -1, -1)
        pad = k.size(-1) // 2
        up = F.conv2d(up, k, padding=pad)
    return up.clamp(0, 1)
