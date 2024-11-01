import torch
import torch.nn.functional as F

def calculate_ssim(img1, img2, window_size=11, max_pixel_value=1.0, K1=0.01, K2=0.03):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    
    Args:
    - img1: The first image (clean image), a PyTorch tensor.
    - img2: The second image (predicted or degraded image), a PyTorch tensor.
    - window_size: Size of the Gaussian window, default is 11.
    - max_pixel_value: Maximum possible pixel value (default is 1.0 if images are normalized).
    - K1, K2: Constants to stabilize the division (default values are 0.01 and 0.03).
    
    Returns:
    - ssim: The SSIM value as a float.
    """
    # Define constants for SSIM
    C1 = (K1 * max_pixel_value) ** 2
    C2 = (K2 * max_pixel_value) ** 2
    
    # Gaussian kernel for convolution
    window = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
    window = window.to(img1.device)  # Ensure window is on the same device as the images
    
    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=1)
    
    # Compute squares of means
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=1) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = ssim_map.mean()
    
    return ssim.item()


def calculate_psnr(clean, pred, max_pixel_value=1.0):
    mse = torch.mean((clean - pred) ** 2)
    if mse == 0:
        return float('inf')  # Return infinity if there's no error (i.e., perfect reconstruction)
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()