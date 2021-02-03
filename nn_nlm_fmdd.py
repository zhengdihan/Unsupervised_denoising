from models.unet import UNet

import torch

from PIL import Image
import numpy as np

from skimage.measure import compare_psnr, compare_ssim
from skimage.restoration import estimate_sigma, denoise_nl_means

import matplotlib.pyplot as plt

import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=False)
    
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

def denoising(noise_im, clean_im, LR=1e-2, sigma=5, rho=1, eta=0.5, alpha=1,
              total_step=20, prob1_iter=500, noise_level=None, result_root=None, f=None):
    input_depth = 1
    latent_dim = 1
    
    en_net = UNet(input_depth, latent_dim, need_sigmoid = False).cuda()
    de_net = UNet(latent_dim, input_depth, need_sigmoid = False).cuda()
        
    parameters = [p for p in en_net.parameters()] + [p for p in de_net.parameters()]                 
    optimizer = torch.optim.Adam(parameters, lr=LR)
        
    l2_loss = torch.nn.MSELoss().cuda()

    i0 = torch.Tensor(noise_im)[None, None, ...].cuda()    
    noise_im_torch = torch.Tensor(noise_im)[None, None, ...].cuda()
    Y = torch.zeros_like(noise_im_torch).cuda()
    i0_til_torch = torch.Tensor(noise_im)[None, None, ...].cuda()
    
    diff_original_np = noise_im.astype(np.float32) - clean_im.astype(np.float32)
    diff_original_name = 'Original_dis.png'
    save_hist(diff_original_np, result_root+diff_original_name)
    
    best_psnr = 0
    best_ssim = 0
    
    for i in range(total_step):

################################# sub-problem 1 ###############################

        for i_1 in range(prob1_iter):
            mean_i = en_net(noise_im_torch)
        
            eps = mean_i.clone().normal_()
            out = de_net(mean_i + eps)
        
            total_loss = 0.5 * l2_loss(out, noise_im_torch)
            total_loss += 1/(2 * sigma**2) * l2_loss(mean_i, i0)
            total_loss += (rho/2) * l2_loss(i0 + Y, i0_til_torch)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                i0 = ((1/sigma**2)*mean_i + rho*(i0_til_torch - Y) + alpha*noise_im_torch) / ((1/sigma**2) + rho + alpha)
        
        with torch.no_grad():
            
################################# sub-problem 2 ###############################
            
            i0_np = i0.cpu().squeeze().detach().numpy()
            Y_np = Y.cpu().squeeze().detach().numpy()
            sig = noise_level
            
            i0_til_np = denoise_nl_means(i0_np + Y_np, h=20*sig, sigma=sig,
                                         fast_mode=False, **patch_kw)  
            
            i0_til_torch = torch.Tensor(i0_til_np[None, None, ...]).cuda()
    
################################# sub-problem 3 ###############################
            
            Y = Y + eta * (i0 - i0_til_torch)
            
###############################################################################            
            
            denoise_obj_pil = Image.fromarray((i0_np + Y_np).clip(0, 255).astype(np.uint8))
            
            Y_np = Y.cpu().squeeze().detach().numpy()
            Y_norm_np = np.abs(Y_np)
            i0_pil = Image.fromarray(np.uint8(i0_np.clip(0, 255)))
            
            mean_i_np = mean_i.cpu().squeeze().detach().numpy().clip(0, 255)
            mean_i_pil = Image.fromarray(mean_i_np.astype(np.uint8))

            out_np = out.cpu().squeeze().detach().numpy().clip(0, 255)
            out_pil = Image.fromarray(out_np.astype(np.uint8))
            
            diff_np = mean_i_np - clean_im
            
            denoise_obj_name = 'denoise_obj_{:04d}'.format(i) + '.png'
            Y_name = 'Y_{:04d}'.format(i) + '.png'
            i0_name = 'i0_num_epoch_{:04d}'.format(i) + '.png'
            mean_i_name = 'Latent_im_num_epoch_{:04d}'.format(i) + '.png'
            out_name = 'res_of_dec_num_epoch_{:04d}'.format(i) + '.png'
            diff_name = 'Latent_dis_num_epoch_{:04d}'.format(i) + '.png'
            
            denoise_obj_pil.save(result_root + denoise_obj_name)
            save_heatmap(Y_norm_np, result_root + Y_name)
            i0_pil.save(result_root + i0_name)
            mean_i_pil.save(result_root + mean_i_name)
            out_pil.save(result_root + out_name)
            save_hist(diff_np, result_root + diff_name)
            
            i0_til_np = i0_til_torch.cpu().squeeze().detach().numpy()
            i0_til_np = np.clip(i0_til_np, 0, 255)
                        
            psnr = compare_psnr(clean_im, i0_til_np, data_range=255)
            ssim = compare_ssim(clean_im, i0_til_np, data_range=255)
            
            i0_til_pil = Image.fromarray(i0_til_np.astype(np.uint8))
            i0_til_pil.save(os.path.join(result_root, '{}'.format(i) + '.png'))

            print('Iteration: {:02d}, VAE Loss: {:f}, PSNR: {:f}, SSIM: {:f}'.format(i, total_loss.item(), psnr, ssim), file=f, flush=True)
            
            if best_psnr < psnr:
                best_psnr = psnr
                best_ssim = ssim
            else:
                break
            
    return i0_til_np, best_psnr, best_ssim


###############################################################################


if __name__ == "__main__":
    path = './data/FMDD/'
    
    noises = sorted(glob.glob(path + 'raw' + '/*.png'))
    cleans = sorted(glob.glob(path + 'gt' + '/*.png'))
    
    LR = 1e-1
    sigma = 1
    rho = 1
    eta = 1
    alpha = 2
    prob1_iter = 10
    total_step = 50
    
    psnrs = []
    ssims = []
    
    for noise, clean, in zip(noises, cleans):
        result_root = './output/nn_nlm_fmdd/{}/'.format(noise.split('/')[-1][:-4])
        os.system('mkdir -p ' + result_root)
        
        noise_im = Image.open(noise)
        clean_im = Image.open(clean)
        
        noise_im_np = np.array(noise_im).astype(np.float)
        clean_im_np = np.array(clean_im).astype(np.float)
        
        noise_level = estimate_sigma(noise_im_np)
        
        with open(result_root + 'result.txt', 'w') as f:
            _, psnr, ssim = denoising(noise_im_np, clean_im_np, LR=LR, sigma=sigma, rho=rho, eta=eta, alpha=alpha,
                                      total_step=total_step, prob1_iter=prob1_iter, 
                                      noise_level=noise_level, result_root=result_root, f=f)
            
            psnrs.append(psnr)
            ssims.append(ssim)
            
    with open('./output/nn_nlm_fmdd/' + 'psnr_ssim.txt', 'w') as f:
        print('PSNR: {}'.format(sum(psnrs)/len(psnrs)), file=f)
        print('SSIM: {}'.format(sum(ssims)/len(ssims)), file=f)
