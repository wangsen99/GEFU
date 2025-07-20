import math
import cv2
import numpy as np
import lpips
import torch
import os
from tqdm import tqdm

#### PSNR
def img_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

#### SSIM
def img_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

#### LPIPS
loss_fn = lpips.LPIPS(net='alex', spatial=False).cuda()
def img_lpips(img1, img2):
    def process(img):
        img = torch.from_numpy(img)[:,:,[2,1,0]].float()
        return img.permute(2,0,1).unsqueeze(0).cuda() * 2 - 1
    img1 = process(img1)
    img2 = process(img2)
    return loss_fn.forward(img1, img2).mean().detach().cpu().tolist()

def metric(gt_image_path, pred_image_path):
    gt_image = cv2.imread(gt_image_path) / 255.
    pred_image = cv2.imread(pred_image_path) / 255.
    pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))

    psnr = img_psnr(gt_image, pred_image)
    ssim = img_ssim(gt_image, pred_image)
    lpips = img_lpips(gt_image, pred_image)
    return psnr, ssim, lpips

def process_image_pairs(real_images_dir, enhanced_images_dir):
    real_image_files = os.listdir(real_images_dir)
    enhanced_image_files = os.listdir(enhanced_images_dir)

    sorted(real_image_files)
    sorted(enhanced_image_files)

    assert len(real_image_files) == len(enhanced_image_files)

    total_psnr = 0
    total_ssim = 0
    total_lpips = 0

    for real_image_file, enhanced_image_file in tqdm(zip(real_image_files, enhanced_image_files)):
        real_image_path = os.path.join(real_images_dir, real_image_file)
        enhanced_image_path = os.path.join(enhanced_images_dir, enhanced_image_file)
        psnr, ssim, lpips_value = metric(real_image_path, enhanced_image_path)
        print(f"{real_image_file} vs {enhanced_image_file}: PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips_value:.4f}")

        total_psnr += psnr
        total_ssim += ssim
        total_lpips += lpips_value

    num_images = len(real_image_files)
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_lpips = total_lpips / num_images

    print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips:.4f}")

real_images_dir = ""
enhanced_images_dir = ""
process_image_pairs(real_images_dir, enhanced_images_dir)