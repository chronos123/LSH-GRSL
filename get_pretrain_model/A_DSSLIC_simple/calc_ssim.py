try:
    from skimage.measure import compare_ssim
except:
    from skimage.metrics import structural_similarity as compare_ssim

from PIL import Image
import numpy as np
from pytorch_msssim import ms_ssim
import torch


def _calc_ssim_array(im1, im2):
    return compare_ssim(im1, im2, multichannel=True, data_range=255)


def _calc_ssim_dir(im_dir1, im_dir2):
    im1 = Image.open(im_dir1)
    im2 = Image.open(im_dir2)
    return compare_ssim(np.asarray(im1), np.asarray(im2), multichannel=True)    

def _calc_ms_ssim_array(im1, im2):
    im1 = np.transpose(im1, (2, 0, 1))
    im2 = np.transpose(im2, (2, 0, 1))
    im1 = torch.from_numpy(im1)
    im2 = torch.from_numpy(im2)
    im1.unsqueeze_(0)
    im2.unsqueeze_(0)
    return ms_ssim(im1, im2, data_range=255)


def _calc_ms_ssim_dir(im_dir1, im_dir2):
    im1 = np.asarray(Image.open(im_dir1)).astype(np.float32)
    im2 = np.asarray(Image.open(im_dir2)).astype(np.float32)
    return _calc_ms_ssim_array(im1, im2)


if __name__ == "__main__":
    dir1 = raw_input("dir1: ")
    dir2 = raw_input("dir2: ")
    print("ssim is {0:.4f}".format(_calc_ssim_dir(dir1, dir2)))
    