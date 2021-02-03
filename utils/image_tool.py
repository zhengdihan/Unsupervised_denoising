from PIL import Image
import torch
import numpy as np

def pil_to_np(img_pil, normalize=False):
    img_np = np.array(img_pil)
    if normalize:
        img_np = img_np / 255
        
    if len(img_np.shape) == 2:
        return img_np.astype(np.float)
    else:
        return img_np.transpose(2, 0, 1).astype(np.float)
    
def np_to_pil(img_np, normalize=False):
    if normalize:
        img_np = img_np*255
    
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    
    if len(img_np.shape) == 2:
        img_pil = Image.fromarray(img_np)
    else:
        img_pil = Image.fromarray(img_np.transpose(1, 2, 0))
        
    return img_pil

def np_to_torch(img_np):
    if len(img_np.shape) == 2:
        return torch.Tensor(img_np)[None, None, ...]
    else:
        return torch.Tensor(img_np)[None, ...]

def torch_to_np(img_torch):
    return img_torch.cpu().squeeze().detach().numpy()
