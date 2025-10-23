
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import color
import numpy as np
def calculate_metrics(pred, target, input_img):
    # Convert tensors to numpy arrays and transpose to (H,W,C)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    input_img = input_img.cpu().numpy()
    
    # Convert from (C,H,W) to (H,W,C)
    pred = np.transpose(pred, (1,2,0))
    target = np.transpose(target, (1,2,0))
    input_img = np.transpose(input_img, (1,2,0))

    # Ensure values are in [0,1] range
    pred = pred / 255.0 if pred.max() > 1.0 else pred
    target = target / 255.0 if target.max() > 1.0 else target
    input_img = input_img / 255.0 if input_img.max() > 1.0 else input_img

    # Convert to grayscale
    pred_gray = color.rgb2gray(pred)
    target_gray = color.rgb2gray(target)
    input_gray = color.rgb2gray(input_img)

    # Compute metrics
    pred_psnr = psnr(target_gray, pred_gray, data_range=1.0)
    pred_ssim = ssim(target_gray, pred_gray, data_range=1.0)
    
    input_psnr = psnr(target_gray, input_gray, data_range=1.0)
    input_ssim = ssim(target_gray, input_gray, data_range=1.0)
    
    psnr_change = ((pred_psnr - input_psnr) / input_psnr) * 100
    ssim_change = ((pred_ssim - input_ssim) / input_ssim) * 100
    
    return pred_psnr, input_psnr, psnr_change, pred_ssim, input_ssim, ssim_change

def calculate_metrics_sc(pred, target, input_img):
    # Convert tensors to numpy arrays
    pred = pred.cpu().numpy().squeeze()  # Remove channel dimension
    target = target.cpu().numpy().squeeze()
    input_img = input_img.cpu().numpy().squeeze()

    # Ensure values are in [0,1] range
    pred = pred / 255.0 if pred.max() > 1.0 else pred
    target = target / 255.0 if target.max() > 1.0 else target
    input_img = input_img / 255.0 if input_img.max() > 1.0 else input_img

    # Compute metrics directly on grayscale images
    pred_psnr = psnr(target, pred, data_range=1.0)
    pred_ssim = ssim(target, pred, data_range=1.0)
    
    input_psnr = psnr(target, input_img, data_range=1.0)
    input_ssim = ssim(target, input_img, data_range=1.0)
    
    psnr_change = ((pred_psnr - input_psnr) / input_psnr) * 100
    ssim_change = ((pred_ssim - input_ssim) / input_ssim) * 100
    
    return pred_psnr, input_psnr, psnr_change, pred_ssim, input_ssim, ssim_change