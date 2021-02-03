import numpy as np

def NoiseEstimation(im_noisy, PatchSize):
    p_out = image2cols(im_noisy, PatchSize, 3)
    
    mu = np.mean(p_out, 1, keepdims=True)
    sigma = np.matmul(p_out - np.tile(mu, (1, p_out.shape[1])), 
                      (p_out - np.tile(mu, (1, p_out.shape[1]))).T) / p_out.shape[1]
    
    eigvalue, _ = np.linalg.eig(sigma)
    eigvalue = np.sort(eigvalue)
    
    for CompCnt in range(p_out.shape[0])[::-1]:
        Mean = np.mean(eigvalue[:CompCnt])
        if np.sum(eigvalue[:CompCnt] > Mean) == np.sum(eigvalue[:CompCnt] < Mean):
            break
        
    return np.sqrt(Mean)

def image2cols(im, pSz, stride):
    range_y = [i for i in range(0, im.shape[0] - pSz+1, stride)]
    range_x = [i for i in range(0, im.shape[1] - pSz+1, stride)]
    channel = im.shape[2]
    
    if range_y[-1] != im.shape[0] - pSz:
        range_y = range_y + [im.shape[0] - pSz]
    if range_x[-1] != im.shape[1] - pSz:
        range_x = range_x + [im.shape[1] - pSz]
    sz = len(range_y) * len(range_x)
    
    tmp = np.zeros(((pSz**2)*channel, sz))
    
    idx = 0
    for y in range_y:
        for x in range_x:
            p = im[y:y+pSz, x:x+pSz, :]
            tmp[:, idx] = p.flatten()
            idx += 1
            
    return tmp

if __name__ == '__main__':
    a = np.random.randn(512, 512, 3)*25
    sigma = NoiseEstimation(a, 7)
    print(sigma)
