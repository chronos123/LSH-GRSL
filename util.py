from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2


def yuv2bgr(img, nbit=10):
      
    minus = 2 ** (nbit - 1)
    
    y = img[:,:,0].astype(np.float32)
    u = img[:,:,1].astype(np.float32)
    v = img[:,:,2].astype(np.float32)

    r = np.round(y + 1.13983 * (v-minus))

    g = np.round(y - 0.39465 * (u-minus) - 0.58060 * (v-minus))

    b = np.round(y + 2.03211 * (u-minus))

    return np.stack((b, g, r), axis=2).astype(np.uint16)


def bgr2yuv(img, nbit=10):
      
    add = 2 ** (nbit - 1)
    
    b = img[:,:,0].astype(np.float32)
    g = img[:,:,1].astype(np.float32)
    r = img[:,:,2].astype(np.float32)

    y = np.round(0.299*r + 0.587*g + 0.114*b)
    u = np.round(- 0.148*r - 0.289*g + 0.437*b) + add
    v = np.round(0.615*r - 0.515*g - 0.1*b) + add

    return np.stack((y, u, v), axis=2).astype(np.uint16)


def save_res(image_array, save_path, nbit=10):
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2YUV)
    image_array = bgr2yuv(image_array, nbit=nbit)
    img_bytes = image_array.tobytes()
    # image_array = np.where()
    with open(save_path, "wb") as f:
        f.write(image_array)
    # cv2.imwrite(save_path, image_array)
    

def get_res(save_path, nbit=10):
    # yuv = cv2.imread(save_path, -1)
    with open(save_path, "rb") as f:
        yuv = f.read()
        img_1 = np.frombuffer(yuv, np.uint16)
        yuv = np.reshape(img_1, (1152, 1600, 3))
        # yuv = cv2.imdecode(yuv, cv2.IMREAD_ANYCOLOR)
        
    # bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    bgr = yuv2bgr(yuv, nbit=nbit)
    return bgr.astype(np.float32)


# def planertopack(img):
#     """_summary_
#         used to save a yuv file in a packed mode: y1 u1 v1 y2 u2 v2
#         not y1 y2 y3 y4 ... u1 u2 u3 u4 .... v1 v2 v3 v4 ....
#     Args:
#         img (_type_): _description_
#     """
#     pass


def save_image_my(image_array, save_path):
    image_array = image_array.astype(np.uint8)
    Image.fromarray(image_array).save(save_path)


def get_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    #print(image_tensor)
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

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    #if label_tensor==0:
    #	return 0;
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

    