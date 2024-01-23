import torch
import numpy as np
from PIL import Image
import os
import cv2
import math


def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    #print(image_tensor)
    if len(image_tensor.shape) == 4:
        image_numpy = []
        for i in range(image_tensor.shape[0]):
            image_numpy.append(tensor2im(image_tensor[i, :, :, :], imtype, normalize))
        return image_numpy
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


def psnr_array(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr_batch_all(data1, data2, norm=True):
    images1 = tensor2im(data1.data, normalize=norm)
    images2 = tensor2im(data2.data, normalize=norm)
    psnr = 0
    for i, t in enumerate(zip(images1, images2)):
        im1, im2 = t
        psnr += psnr_array(im1, im2)    
        
    return psnr


def filesize(filepath):
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError("Invalid file {0}.".format(filepath))
    file_stat =  os.stat(filepath)
    return file_stat.st_size


def calc_bpp_vvc_exp(file, choice="mars"):
    if choice == "mars":
        height = 1152
        width = 1600
    else:
        raise ValueError()
    
    bpp = filesize(file) * 8.0 / (height * width)
    return bpp


def psnr_10bit(img1: np.ndarray, img2: np.ndarray):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1/1024. - img2/1024.) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 10 * math.log10(PIXEL_MAX * PIXEL_MAX/ mse)


def psnr_10bit_dir(src, dst):
    src_im = cv2.imread(src, cv2.IMREAD_ANYDEPTH)
    dst_im = cv2.imread(dst, cv2.IMREAD_ANYDEPTH)
    return psnr_10bit(src_im, dst_im)
