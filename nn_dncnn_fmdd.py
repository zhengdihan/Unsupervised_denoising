from models.unet import UNet
from models.network_dncnn import DnCNN

import torch

from PIL import Image
from utils.image_tool import pil_to_np, np_to_pil, np_to_torch, torch_to_np

import numpy as np

from skimage.measure import compare_psnr, compare_ssim

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

def save_torch(img_torch, root):
    img_np = torch_to_np(img_torch)
    img_pil = np_to_pil(img_np)
    img_pil.save(root)

def iteration_decay(num_iter):
    num_iter = int(num_iter/1.6)
    if num_iter <= 100:
        num_iter = 100
    return num_iter


def denoising_gray(noise_im, clean_im, LR=1e-2, sigma=5, rho=1, eta=0.5, 
                   total_step=20, prob1_iter=1000, result_root=None, fo=None):
    input_depth = 1
    latent_dim = 1
    
    en_net = UNet(input_depth, latent_dim, need_sigmoid = False).cuda()
    de_net = UNet(latent_dim, input_depth, need_sigmoid = False).cuda()
    
    model = DnCNN(1, 1, nc=64, nb=20, act_mode='R')
    model_path = './model_zoo/dncnn_gray_blind.pth'
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.cuda()
    for k, v in model.named_parameters():
        v.requires_grad = False

    en_optimizer = torch.optim.Adam(en_net.parameters(), lr = LR)
    de_optimizer = torch.optim.Adam(de_net.parameters(), lr = LR)
    
    l2_loss = torch.nn.MSELoss().cuda()
    
    i0 = np_to_torch(noise_im).to(device)
    noise_im_torch = np_to_torch(noise_im).to(device)
    Y = torch.zeros_like(noise_im_torch).to(device)
    i0_til_torch = np_to_torch(noise_im).to(device)
    
    best_psnr = 0
    best_ssim = 0
    
    for i in range(total_step):
        
############################### sub-problem 1 #################################

        prob1_iter = iteration_decay(prob1_iter)

        for i_1 in range(prob1_iter):
            mean = en_net(noise_im_torch)

            eps = mean.clone().normal_()
            out = de_net(mean + eps)

            total_loss =  0.5 * l2_loss(out, noise_im_torch)
            total_loss += 0.5 * 1/(sigma**2) * l2_loss(mean, i0)
            
            en_optimizer.zero_grad()
            de_optimizer.zero_grad()

            total_loss.backward()

            en_optimizer.step()
            de_optimizer.step()
            
            with torch.no_grad():
                i0 = ((1/sigma**2)*mean + rho*(i0_til_torch - Y)) / ((1/sigma**2) + rho)
        
        with torch.no_grad():
            
############################### sub-problem 2 #################################
            
            i0_til_torch = model((i0+Y)/255) * 255

############################### sub-problem 3 #################################

            Y = Y + eta * (i0 - i0_til_torch)

###############################################################################

            i0_til_np = torch_to_np(i0_til_torch).clip(0, 255)            
            psnr_gt = compare_psnr(clean_im, i0_til_np, 255)
            ssim_gt = compare_ssim(i0_til_np, clean_im, multichannel=False, data_range=255)
            
            denoise_obj_name = 'denoise_obj_{:04d}'.format(i) + '.png'            
            i0_name = 'i0_num_epoch_{:04d}'.format(i) + '.png'
            result_name = 'num_epoch_{:04d}'.format(i) + '.png'
            mean_name = 'Latent_im_num_epoch_{:04d}'.format(i) + '.png'
            out_name = 'res_of_dec_num_epoch_{:04d}'.format(i) + '.png'
            
            save_torch(Y+i0, result_root + denoise_obj_name)
            save_torch(i0, result_root + i0_name)
            save_torch(i0_til_torch, result_root + result_name)
            save_torch(mean, result_root + mean_name)
            save_torch(out, result_root + out_name)

            print('Iteration %02d  Loss %f  PSNR_gt: %f, SSIM_gt: %f' % (i, total_loss.item(), psnr_gt, ssim_gt), file=fo, flush=True)
            
            if best_psnr < psnr_gt:
                best_psnr = psnr_gt
                best_ssim = ssim_gt
            else:
                break
    
    return best_psnr, best_ssim


###############################################################################


if __name__ == "__main__":
    path = './data/FMDD/'
    
    noises = sorted(glob.glob(path + 'raw' + '/*.png'))
    cleans = sorted(glob.glob(path + 'gt' + '/*.png'))

    LR = 1e-2
    sigma = 0.1
    rho = 1
    eta = 0.5
    total_step = 20
    prob1_iter = 1000
    
    psnrs = []
    ssims = []
    
    for clean_root, noise_root in zip(cleans, noises):
        result = './output/nn_dncnn_fmdd/{}/'.format(noise_root.split('/')[-1][:-4])
        os.system('mkdir -p ' + result)
        
        noise_im = Image.open(noise_root)
        clean_im = Image.open(clean_root)
        noise_im_np = pil_to_np(noise_im)
        clean_im_np = pil_to_np(clean_im)
        
        with open(result + '/result.txt', 'w') as fo:
            psnr_ssim = denoising_gray(noise_im_np, clean_im_np, LR=LR, sigma=sigma, 
                                       rho=rho, eta=eta, total_step=total_step, 
                                       prob1_iter=prob1_iter, result_root=result, fo=fo)

        psnrs.append(psnr_ssim[0])
        ssims.append(psnr_ssim[1])
    
    with open('./output/nn_dncnn_fmdd/' + 'psnr_ssim.txt', 'w') as f:
        print('PSNR: {}'.format(sum(psnrs)/len(psnrs)), file=f)
        print('SSIM: {}'.format(sum(ssims)/len(ssims)), file=f)
