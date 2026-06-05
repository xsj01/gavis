#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

if __name__ == "__main__":
    def test_psnr():
        batched_img = torch.rand(10, 512, 512, 3)
        noisy_img = batched_img + 0.05 * torch.randn_like(batched_img)
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        psnr_vals = psnr(batched_img, noisy_img)
        print(psnr_vals.mean().item())

        psnr_list = []
        for i in range(batched_img.shape[0]):
            single_psnr = psnr(batched_img[i:i+1], noisy_img[i:i+1])
            psnr_list.append(single_psnr)
        single_psnr = torch.stack(psnr_list, dim=0)
        print(single_psnr.mean().item())
    test_psnr()