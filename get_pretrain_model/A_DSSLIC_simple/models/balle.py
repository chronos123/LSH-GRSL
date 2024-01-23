from .InvCompress.codes.compressai.models.priors import ScaleHyperprior
from .InvCompress.codes.examples.train import RateDistortionLoss, AverageMeter

"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """
    
## model has the aus_loss method, we can write the training code according to the 
# train-one-epoch method the same file as AverageMeter

# lr: 1e-4
# N = 128 , M = 192 ---------- low lambda
# N = 192, M = 320 ------------- high lambda
# lambda in RateDistortionLoss(default 1e-2), 选择不同的lambda值来控制 R + lambda * D
# lambda 越高则码率越高，R-D优化问题
