from models.unet import UNet

import torch

from PIL import Image
import numpy as np

import bm3d
from skimage.measure import compare_psnr, compare_ssim

import matplotlib.pyplot as plt

import glob

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def save_hist(x, root):
    x = x.flatten()
    plt.figure()
    n, bins, patches = plt.hist(x, bins=128, density=1)
    plt.savefig(root)
    plt.close()
    
def save_heatmap(image_np, root):
    cmap = plt.get_cmap('jet')
    
    rgba_img = cmap(image_np)
    rgb_img = np.delete(rgba_img, 3, 2)
    rgb_img_pil = Image.fromarray((255*rgb_img).astype(np.uint8))
    rgb_img_pil.save(root)

def denoising_fixd_noise_level(noise_im, clean_im, LR=1e-2, sigma=5, rho=1, eta=0.5, 
                               alpha=0.5, total_step=19, prob1_iter=1000, 
                               noise_level=None, result_root=None, fo=None):
    sig = noise_level
    r_bm3d = bm3d.bm3d_rgb(noise_im.transpose(1, 2, 0), sig)
    r_bm3d = np.clip(r_bm3d, 0, 255)
    psnr_bm3d = compare_psnr(clean_im.transpose(1, 2, 0), r_bm3d, 255)
    ssim_bm3d = compare_ssim(r_bm3d, clean_im.transpose(1, 2, 0), multichannel=True, data_range=255)

    print('noise level {} '.format(noise_level), file=fo, flush=True)
    print('PSNR_BM3D: {}, SSIM_BM3D: {}'.format(psnr_bm3d, ssim_bm3d), file=fo, flush=True)
    
    r_bm3d = Image.fromarray(r_bm3d.astype(np.uint8))
    r_bm3d.save(result_root + 'bm3d_result.png')
    
    input_depth = 3
    latent_dim = 3

    en_net = UNet(input_depth, latent_dim, need_sigmoid=True).cuda()
    de_net = UNet(latent_dim, input_depth, need_sigmoid=True).cuda()
    
    en_optimizer = torch.optim.Adam(en_net.parameters(), lr = LR)
    de_optimizer = torch.optim.Adam(de_net.parameters(), lr = LR)
    
    l2_loss = torch.nn.MSELoss().cuda()
    
    noise_im = noise_im / 255.0
    i0 = torch.Tensor(noise_im[None, ...]).cuda()
    noise_im_torch = torch.Tensor(noise_im)[None, ...].cuda()
    Y = torch.zeros_like(noise_im_torch).cuda()
    i0_til_torch = torch.Tensor(noise_im[None, ...]).cuda()
    
    output = None

    for i in range(total_step):
        
############################### sub-problem 1 #################################

        for i_1 in range(prob1_iter):
            mean_i = en_net(noise_im_torch)

            eps = mean_i.clone().normal_()
            out = de_net(mean_i + eps)

            total_loss = 0.5 * l2_loss(out, noise_im_torch)
            total_loss += 1/(2 * sigma**2) * l2_loss(mean_i, i0)
            
            en_optimizer.zero_grad()
            de_optimizer.zero_grad()

            total_loss.backward()

            en_optimizer.step()
            de_optimizer.step()
            
            with torch.no_grad():
                i0 = ((1/sigma**2)*mean_i + rho*(i0_til_torch - Y) + alpha*noise_im_torch) / ((1/sigma**2) + rho + alpha)
        
        with torch.no_grad():
            
############################### sub-problem 2 #################################
            
            i0_np = i0.cpu().squeeze().detach().numpy()
            Y_np = Y.cpu().squeeze().detach().numpy()
            
            tmp = i0_np.transpose(1, 2, 0) + Y_np.transpose(1, 2, 0)
            tmp = np.clip(tmp, 0, 1)
            
            sig = noise_level
            
            i0_til_np = bm3d.bm3d_rgb(tmp*255, sig) / 255
            i0_til_torch = torch.Tensor(i0_til_np.transpose(2, 0, 1)[None, ...]).cuda()

############################### sub-problem 3 #################################

            Y = Y + eta * (i0 - i0_til_torch)

###############################################################################

            i0_til_np = i0_til_torch.cpu().squeeze().detach().numpy()
            i0_til_np = np.clip(i0_til_np, 0, 1)
            output = i0_til_np

            psnr_gt = compare_psnr(clean_im.transpose(1, 2, 0), 255*i0_til_np.transpose(1, 2, 0), 255)
            ssim_gt = compare_ssim(255*i0_til_np.transpose(1, 2, 0), clean_im.transpose(1, 2, 0) , multichannel=True, data_range=255)

            if not i % 5:

                denoise_obj_pil = Image.fromarray((tmp*255).astype(np.uint8))                
                Y_np = Y.cpu().squeeze().detach().numpy()                
                i0_np = np.clip(i0_np, 0, 1)
                i0_pil = Image.fromarray(np.uint8(255*i0_np.transpose(1, 2, 0)))                
                i0_til_np = i0_til_np.transpose(1, 2, 0)
                i0_til_pil = Image.fromarray((255*i0_til_np).astype(np.uint8))
                mean_i_np = mean_i.cpu().squeeze().detach().numpy().clip(0, 1)
                mean_i_pil = Image.fromarray((255*mean_i_np.transpose(1, 2, 0)).astype(np.uint8))
                out_np = out.cpu().squeeze().detach().numpy().clip(0, 1)
                out_pil = Image.fromarray((255*out_np.transpose(1, 2, 0)).astype(np.uint8))
                
                denoise_obj_name = 'denoise_obj_{:04d}'.format(i) + '.png'
                Y_name = 'Y_{:04d}'.format(i) + '.png'
                i0_name = 'i0_num_epoch_{:04d}'.format(i) + '.png'
                result_name = 'num_epoch_{:04d}'.format(i) + '.png'
                mean_i_name = 'Latent_im_num_epoch_{:04d}'.format(i) + '.png'
                out_name = 'res_of_dec_num_epoch_{:04d}'.format(i) + '.png'
                
                denoise_obj_pil.save(result_root + denoise_obj_name)                
                Y_norm_np = np.sqrt((Y_np*Y_np).sum(0))
                save_heatmap(Y_norm_np, result_root + Y_name)
                i0_pil.save(result_root + i0_name)
                i0_til_pil.save(result_root + result_name)
                mean_i_pil.save(result_root + mean_i_name)
                out_pil.save(result_root + out_name)

            print('Iteration %02d  Loss %f  PSNR_gt: %f, SSIM_gt: %f' % (i, total_loss.item(), psnr_gt, ssim_gt), file=fo, flush=True)

    psnr = psnr_gt
    ssim = ssim_gt
    
    output_pil = Image.fromarray((255*output.transpose(1, 2, 0)).astype(np.uint8))
    output_pil.save(result_root + 'ours_result.png')

    return psnr, ssim, psnr_bm3d, ssim_bm3d


###############################################################################


if __name__ == "__main__":
    path = './data/set9/'
    cleans = sorted(glob.glob(path+'*.png'))
    noises = [25, 50, 75, 100]
    
    LR = 1e-1
    rho = 1
    eta = 0.1
    alpha = 1
    prob1_iter = 10
    
    for noise in noises:
        if noise == 25:
            sigma = 15
        else:
            sigma = 3
            
        our_psnr = []
        our_ssim = []
        bm3d_psnr = []
        bm3d_ssim = []
        
        for clean in cleans:
            result = './output/nn_bm3d_AWGN_set9/Gaussian_{}/{}/'.format(noise, clean.split('/')[-1][:-4])
            os.system('mkdir -p ' + result)
            clean_im = Image.open(clean)
            clean_im_np = np.array(clean_im).transpose(2, 0, 1)
            noise_im_np = clean_im_np + noise * np.random.randn(clean_im_np.shape[0], 
                                                                clean_im_np.shape[1], 
                                                                clean_im_np.shape[2])
            noise_im_np = np.clip(noise_im_np, 0, 255)
            noise_im_np = (noise_im_np.astype(np.uint8)).astype(np.float)
                        
            with open(result + '/result.txt', 'w') as fo:
                psnr_ssim = denoising_fixd_noise_level(noise_im_np, clean_im_np, LR=LR, sigma=sigma, 
                                               rho=rho, eta=eta, alpha=alpha, total_step=101, 
                                               prob1_iter=prob1_iter, noise_level=noise, 
                                               result_root=result, fo=fo)
                
            with open(result + '/psnr.txt', 'w') as f:
                print(psnr_ssim, file=f, flush=True)

            our_psnr.append(psnr_ssim[0])
            our_ssim.append(psnr_ssim[1])
            bm3d_psnr.append(psnr_ssim[2])
            bm3d_ssim.append(psnr_ssim[3])
        
        result = './output/nn_bm3d_AWGN_set9/Gaussian_{}/'.format(noise)
        with open(result + 'psnr_ssim.txt', 'w') as f:
            print('avr our psnr =', sum(our_psnr)/len(our_psnr), file=f, flush=True)
            print('avr our ssim =', sum(our_ssim)/len(our_ssim), file=f, flush=True)
            print('avr bm3d psnr =', sum(bm3d_psnr)/len(bm3d_psnr), file=f, flush=True)
            print('avr bm3d ssim =', sum(bm3d_ssim)/len(bm3d_ssim), file=f, flush=True)
