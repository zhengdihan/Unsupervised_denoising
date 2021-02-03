from models.unet_new import Encoder, Decoder
from models.network_dncnn import DnCNN as net

import torch

from utils.image_io import pil_to_np, np_to_pil, np_to_torch, torch_to_np

from PIL import Image
import numpy as np

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
    
def kl_loss(mean, log_var, mu, sigma):
    kl = 0.5*((1/(sigma**2))*(mean - mu)**2 + torch.exp(log_var)/(sigma**2) + np.log(sigma**2) - log_var - 1)
    loss = torch.mean(torch.sum(kl, dim=(1, 2, 3)))
    
    return loss

def sample_z(mean, log_var):
    eps = mean.clone().normal_()*torch.exp(log_var/2)
    
    return mean + eps    
    
def denoising(noise_im, clean_im, LR=1e-2, sigma=5, rho=1, eta=0.5, 
              total_step=20, prob1_iter=500, result_root=None, f=None):
    
    input_depth = 3
    latent_dim = 3
    
    en_net = Encoder(input_depth, latent_dim, down_sample_norm='batchnorm', 
                     up_sample_norm='batchnorm').cuda()
    de_net = Decoder(latent_dim, input_depth, down_sample_norm='batchnorm', 
                     up_sample_norm='batchnorm').cuda()
    
    model = net(3, 3, nc=64, nb=20, act_mode='R')
    model_path = './model_zoo/dncnn_color_blind.pth'
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.cuda()
    
    noise_im_torch = np_to_torch(noise_im)
    noise_im_torch = noise_im_torch.cuda()
    
    parameters = [p for p in en_net.parameters()] + [p for p in de_net.parameters()]             
    optimizer = torch.optim.Adam(parameters, lr=LR)    
    l2_loss = torch.nn.MSELoss(reduction='sum').cuda()
    
    i0 = np_to_torch(noise_im).cuda()
    Y = torch.zeros_like(noise_im_torch).cuda()
    i0_til_torch = np_to_torch(noise_im).cuda()

    diff_original_np = noise_im.astype(np.float32) - clean_im.astype(np.float32)
    diff_original_name = 'Original_dis.png'
    save_hist(diff_original_np, result_root+diff_original_name)  
    
    best_psnr = 0
    best_ssim = 0
    
    for i in range(total_step):

############################### sub-problem 1 #################################

        for i_1 in range(prob1_iter):
            
            optimizer.zero_grad()
            
            mean, log_var = en_net(noise_im_torch)
        
            z = sample_z(mean, log_var)
            out = de_net(z)
            
            total_loss = 0.5 * l2_loss(out, noise_im_torch)
            total_loss += kl_loss(mean, log_var, i0, sigma)
            total_loss += (rho/2) * l2_loss(i0 + Y, i0_til_torch)
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                i0 = ((1/sigma**2)*mean + rho*(i0_til_torch - Y)) / ((1/sigma**2) + rho)
        
        with torch.no_grad():
            
############################### sub-problem 2 #################################       
     
            i0_til_torch = model(i0+Y)
    
############################### sub-problem 3 #################################           
 
            Y = Y + eta * (i0 - i0_til_torch)

###############################################################################

            i0_np = torch_to_np(i0)
            Y_np = torch_to_np(Y)
            denoise_obj_pil = np_to_pil((i0_np+Y_np).clip(0,1))
            Y_norm_np = np.sqrt((Y_np*Y_np).sum(0))
            i0_pil = np_to_pil(i0_np)
            mean_np = torch_to_np(mean)
            mean_pil = np_to_pil(mean_np)
            out_np = torch_to_np(out)
            out_pil = np_to_pil(out_np)
            diff_np = mean_np - clean_im
            
            denoise_obj_name = 'denoise_obj_{:04d}'.format(i) + '.png'
            Y_name = 'Y_{:04d}'.format(i) + '.png'
            i0_name = 'i0_num_epoch_{:04d}'.format(i) + '.png'
            mean_i_name = 'Latent_im_num_epoch_{:04d}'.format(i) + '.png'
            out_name = 'res_of_dec_num_epoch_{:04d}'.format(i) + '.png'
            diff_name = 'Latent_dis_num_epoch_{:04d}'.format(i) + '.png'
            
            denoise_obj_pil.save(result_root + denoise_obj_name)
            save_heatmap(Y_norm_np, result_root + Y_name)
            i0_pil.save(result_root + i0_name)
            mean_pil.save(result_root + mean_i_name)
            out_pil.save(result_root + out_name)
            save_hist(diff_np, result_root + diff_name)
            i0_til_np = torch_to_np(i0_til_torch).clip(0, 1)
            
            psnr = compare_psnr(clean_im.transpose(1, 2, 0), i0_til_np.transpose(1, 2, 0), 1)
            ssim = compare_ssim(clean_im.transpose(1, 2, 0), i0_til_np.transpose(1, 2, 0), multichannel=True, data_range=1)
            i0_til_pil = np_to_pil(i0_til_np)
            i0_til_pil.save(os.path.join(result_root, '{}'.format(i) + '.png'))

            print('Iteration: %02d, VAE Loss: %f, PSNR: %f, SSIM: %f' % (i, total_loss.item(), psnr, ssim), file=f, flush=True)
            
            if best_psnr < psnr:
                best_psnr = psnr
                best_ssim = ssim
            else:
                break
            
    return i0_til_np, best_psnr, best_ssim

###############################################################################

if __name__ == "__main__":
    path = 'data/CC/'
    noises = sorted(glob.glob(path + '*real.png'))
    cleans = sorted(glob.glob(path + '*mean.png'))
        
    LR = 1e-2
    sigma = 0.5
    rho = 2
    eta = 0.5
    total_step = 20
    prob1_iter = 1000
    
    psnrs = []
    ssims = []
    
    for noise, clean in zip(noises, cleans):
        result = 'output/nn_dncnn_cc/{}/'.format(noise.split('/')[-1][:-9])
        os.system('mkdir -p ' + result)
        
        noise_im_pil = Image.open(noise)
        clean_im_pil = Image.open(clean)
        
        noise_im_np = pil_to_np(noise_im_pil)
        clean_im_np = pil_to_np(clean_im_pil)
        
        with open(result + 'result.txt', 'w') as f:
            
            _, psnr, ssim = denoising(noise_im_np, clean_im_np, LR=LR, sigma=sigma, rho=rho, eta=eta, 
                                      total_step=total_step, prob1_iter=prob1_iter, result_root=result, f=f)
            psnrs.append(psnr)
            ssims.append(ssim)
            
    with open('output/nn_dncnn_cc/' + 'psnr_ssim.txt', 'w') as f:
        print('PSNR: {}'.format(sum(psnrs)/len(psnrs)), file=f)
        print('SSIM: {}'.format(sum(ssims)/len(ssims)), file=f)

