# Unsupervised_denoising
An Unsupervised deep learning approach for real-world image denoising

Designing an unsupervised image denoising approach in practical applications is
a challenging task due to the complicated data acquisition process. In the realworld case, the noise distribution is so complex that the simplified additive white
Gaussian (AWGN) assumption rarely holds, which significantly deteriorates the
Gaussian denoisers’ performance. To address this problem, we apply a deep neural network that maps the noisy image into a latent space in which the AWGN
assumption holds, and thus any existing Gaussian denoiser is applicable. More
specifically, the proposed neural network consists of the encoder-decoder structure and approximates the likelihood term in the Bayesian framework. Together
with a Gaussian denoiser, the neural network can be trained with the input image
itself and does not require any pre-training in other datasets. Extensive experiments on real-world noisy image datasets have shown that the combination of
neural networks and Gaussian denoisers improves the performance of the original
Gaussian denoisers by a large margin. In particular, the neural network+BM3D
method significantly outperforms other unsupervised denoising approaches and is
competitive with supervised networks such as DnCNN, FFDNet, and CBDNet.
