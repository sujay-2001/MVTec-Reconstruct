import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def calculate_ssim(img1, img2, max_pixel_value=1.0):
    ssim_val = ssim(img1, img2, data_range = max_pixel_value)
    return ssim_val.item()


def calculate_psnr(clean, pred, max_pixel_value=1.0):
    mse = torch.mean((clean - pred) ** 2)
    if mse == 0:
        return float('inf')  # Return infinity if there's no error (i.e., perfect reconstruction)
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()