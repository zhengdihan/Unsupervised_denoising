from models.unet import UNet

import torch

from PIL import Image
import numpy as np

from utils.image_tool import pil_to_np, np_to_pil, np_to_torch, torch_to_np

import bm3d
from skimage.measure import compare_psnr, compare_ssim
from skimage.restoration import estimate_sigma

import matplotlib.pyplot as plt

import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def sample_z(mean): 
    eps = mean.clone().normal_()
    
    return mean + eps

def eval_sigma(num_iter, noise_level):
    if num_iter == 1:
        sigma = noise_level
    else:
        sigma = 5
        
    return sigma

def save_torch(img_torch, root):
    img_np = torch_to_np(img_torch)
    img_pil = np_to_pil(img_np)
    img_pil.save(root)

def denoising(noise_im, clean_im, LR=1e-2, sigma=3, rho=1, eta=0.5, total_step=30, 
              prob1_iter=500, noise_level=None, result_root=None, f=None):
    
    input_depth = 3
    latent_dim = 3
    
    en_net = UNet(input_depth, latent_dim).to(device)
    de_net = UNet(latent_dim, input_depth).to(device)
    
    parameters = [p for p in en_net.parameters()] + [p for p in de_net.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=LR)
    
    l2_loss = torch.nn.MSELoss().cuda()
    
    i0 = np_to_torch(noise_im).to(device)
    noise_im_torch = np_to_torch(noise_im).to(device)
    i0_til_torch = np_to_torch(noise_im).to(device)
    Y = torch.zeros_like(noise_im_torch).to(device)
        
    diff_original_np = noise_im.astype(np.float32) - clean_im.astype(np.float32)
    diff_original_name = 'Original_dis.png'
    save_hist(diff_original_np, result_root+diff_original_name)  
    
    best_psnr = 0
    
    for i in range(total_step):
        
################################# sub-problem 1 ###############################

        for i_1 in range(prob1_iter):
            
            optimizer.zero_grad()

            mean = en_net(noise_im_torch)
            z = sample_z(mean)
            out = de_net(z)
            
            total_loss =  0.5 * l2_loss(out, noise_im_torch)
            total_loss += 0.5 * (1/sigma**2)*l2_loss(mean, i0)
            total_loss += (rho/2) * l2_loss(i0 + Y, i0_til_torch)
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                i0 = ((1/sigma**2)*mean.detach() + rho*(i0_til_torch - Y)) / ((1/sigma**2) + rho)
        
        with torch.no_grad():
            
################################# sub-problem 2 ###############################
            
            i0_np = torch_to_np(i0)
            Y_np = torch_to_np(Y)
            
            sig = eval_sigma(i+1, noise_level)
            
            i0_til_np = bm3d.bm3d_rgb(i0_np.transpose(1, 2, 0) + Y_np.transpose(1, 2, 0), sig).transpose(2, 0, 1)
            i0_til_torch = np_to_torch(i0_til_np).to(device)
            
################################# sub-problem 3 ###############################

            Y = Y + eta * (i0 - i0_til_torch)

###############################################################################

            Y_name = 'Y_{:04d}'.format(i) + '.png'
            i0_name = 'i0_num_epoch_{:04d}'.format(i) + '.png'
            mean_name = 'Latent_im_num_epoch_{:04d}'.format(i) + '.png'
            out_name = 'res_of_dec_num_epoch_{:04d}'.format(i) + '.png'
            diff_name = 'Latent_dis_num_epoch_{:04d}'.format(i) + '.png'
            
            Y_np = torch_to_np(Y)
            Y_norm_np = np.sqrt((Y_np*Y_np).sum(0))
            save_heatmap(Y_norm_np, result_root + Y_name)
            
            save_torch(mean, result_root + mean_name)
            save_torch(out, result_root + out_name)
            save_torch(i0, result_root + i0_name)
            
            mean_np = torch_to_np(mean)
            diff_np = mean_np - clean_im
            save_hist(diff_np, result_root + diff_name)

            i0_til_np = torch_to_np(i0_til_torch).clip(0, 255)
            psnr = compare_psnr(clean_im.transpose(1, 2, 0), i0_til_np.transpose(1, 2, 0), 255)
            ssim = compare_ssim(clean_im.transpose(1, 2, 0), i0_til_np.transpose(1, 2, 0), multichannel=True, data_range=255)
            
            i0_til_pil = np_to_pil(i0_til_np)
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
    path = './data/PolyU/'
    noises = sorted(glob.glob(path + '*real.JPG'))
    cleans = sorted(glob.glob(path + '*mean.JPG'))
    
    LR = 1e-2
    sigma = 3
    rho = 1
    eta = 0.5
    total_step = 30
    prob1_iter = 500
    
    psnrs = []
    ssims = []
    
    for noise, clean in zip(noises[96:], cleans[96:]):
        result = './output/nn_bm3d_polyu/{}/'.format(noise.split('/')[-1][:-9])
        os.system('mkdir -p ' + result)
        
        noise_im = Image.open(noise)
        clean_im = Image.open(clean)
        
        noise_im_np = pil_to_np(noise_im)
        clean_im_np = pil_to_np(clean_im)
        
        noise_level = 5
        # noise_level = np.mean(estimate_sigma(noise_im_np.transpose(1, 2, 0), multichannel=True))
        
        with open(result + 'result.txt', 'w') as f:
            _, psnr, ssim = denoising(noise_im_np, clean_im_np, LR=LR, sigma=sigma, 
                                      rho=rho, eta=eta, total_step=total_step, 
                                      prob1_iter=prob1_iter, noise_level=noise_level, 
                                      result_root=result, f=f)

            psnrs.append(psnr)
            ssims.append(ssim)
    with open('./output/nn_bm3d_polyu/' + 'psnr_ssim.txt', 'w') as f:
        print('PSNR: {}'.format(sum(psnrs)/len(psnrs)), file=f)
        print('SSIM: {}'.format(sum(ssims)/len(ssims)), file=f)
