import numpy as np
import math
from util.util import tensor2im
from PIL import Image
import os
import re
import torch
import tqdm
from calc_ssim import _calc_ssim_dir, _calc_ms_ssim_dir
try:
    from skimage.measure import compare_psnr
except:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def yCbCr2rgb(input_im):
    input_im = torch.from_numpy(input_im)
    im_flat = input_im.contiguous().view(-1, 3).float()
    mat = torch.tensor([[1.164, 1.164, 1.164], [0, -0.392, 2.017],
                        [1.596, -0.813, 0]])
    bias = torch.tensor([-16.0 / 255.0, -128.0 / 255.0, -128.0 / 255.0])
    temp = (im_flat + bias).mm(mat)
    out = temp.view(list(input_im.size())[0], list(input_im.size())[1], 3)
    return out.numpy()


def rgb2yCbCr(input_im):
    input_im = torch.from_numpy(input_im)
    im_flat = input_im.contiguous().view(-1, 3).float()
    mat = torch.tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368],
                        [0.098, 0.439, -0.071]])
    bias = torch.tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0])
    temp = im_flat.mm(mat) + bias
    out = temp.view(input_im.shape[0], input_im.shape[1], 3)
    return out.numpy()


def rgb2yuv_array(rgb):
    rgb_ = torch.from_numpy(rgb)
    rgb_ = rgb.transpose(0, 2)  # input is 3*n*n   default
    A = torch.tensor([[0.299, -0.14714119, 0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]])  # from  Wikipedia
    yuv = torch.tensordot(rgb_, A, 1).transpose(0, 2)
    return yuv.numpy()


def yuv2rgb_array(yuv):
    yuv_ = torch.from_numpy(yuv)
    yuv_ = yuv.transpose(0, 2)  # input is 3*n*n   default
    A = torch.tensor([[1., 1., 1.], [0., -0.39465, 2.03211],
                      [1.13983, -0.58060, 0]])  # from  Wikipedia
    rgb = torch.tensordot(yuv_, A, 1).transpose(0, 2)
    return rgb.numpy()


def psnr_tensor(img1, img2):
    raise ValueError("error code")
    img1 = tensor2im(img1)
    img2 = tensor2im(img2)

    mse = np.mean((img1 / 255. - img2 / 255.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr_array_xjy(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.)**2)
    if mse < 1.0e-10:
        return 100
    pixel_max = 1.
    return 20 * math.log10(pixel_max / math.sqrt(mse))


def psnr_array(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr_10bit(img1, img2):
    mse = np.mean((img1 / 1024. - img2 / 1024.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 10 * math.log10(PIXEL_MAX * PIXEL_MAX / mse)


def batch_PSNR(img1, img2, data_range=255):
    Img1 = img1.data.cpu().numpy().astype(np.float32)
    Img2 = img2.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img1.shape[0]):
        PSNR += compare_psnr(Img1[i, :, :, :],
                             Img1[i, :, :, :],
                             data_range=data_range)
    return (PSNR / Img1.shape[0])


def psnr_dir(img1_dir, img2_dir):
    img1 = Image.open(img1_dir).convert("RGB")
    img1 = np.asarray(img1)
    img2 = Image.open(img2_dir).convert("RGB")
    img2 = np.asarray(img2)
    mse = np.mean((img1 / 255. - img2 / 255.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr_255_dir(img1_dir, img2_dir):
    img1 = Image.open(img1_dir).convert("RGB")
    img1 = np.asarray(img1)
    img2 = Image.open(img2_dir).convert("RGB")
    img2 = np.asarray(img2)
    return psnr_array_255(img1, img2)


def psnr_res(img_ori_dir, img_rec_dir):
    global res_max
    global res_min
    img1 = Image.open(img_ori_dir).convert("RGB")
    img1 = np.asarray(img1)
    img2 = Image.open(img_rec_dir).convert("RGB")
    img2 = np.asarray(img2)

    img_res = img1.astype(np.float32) - img2.astype(np.float32)
    # y = (x - min)/(max - min) * 255
    res_max = np.max(np.max(img_res)).astype(np.float32)
    res_min = np.min(np.min(img_res)).astype(np.float32)
    img_res = 255 * (img_res - res_min) / (res_max - res_min)
    img_res = img_res.astype(np.uint8)

    img3 = img2 + img_res

    Image.fromarray(img_res).save("res.png")

    mse = np.mean((img1 / 255. - img3 / 255.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr_res_valid(img_ori_dir, img_up_dir, im_res_dir):
    img1 = Image.open(img_ori_dir).convert("RGB")
    img2 = Image.open(img_up_dir).convert("RGB")

    img1 = np.asarray(img1)
    img2 = np.asarray(img2)

    img_res = Image.open(im_res_dir).convert("RGB")
    img_res = np.asarray(img_res)

    img3 = img2 + img_res

    # Image.fromarray(img3).save("res.png")

    mse = np.mean((img1 / 255. - img3 / 255.)**2)

    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr_res_valid_deal(img_ori_dir, img_up_dir, im_res_dir):
    global res_max
    global res_min

    img1 = Image.open(img_ori_dir).convert("RGB")
    img2 = Image.open(img_up_dir).convert("RGB")

    img1 = np.asarray(img1)
    img2 = np.asarray(img2)

    img_res = Image.open(im_res_dir).convert("RGB")
    img_res = np.asarray(img_res)

    img_res = img_res.astype(np.float32) * (res_max - res_min) / 255 + res_min

    img3 = img2 + img_res.astype(np.uint8)

    # img3 = rgb2yCbCr(img2/255. + img_res/225.)

    # img3 = yCbCr2rgb(img3) * 255
    # img3 = img3.astype(np.uint8)

    Image.fromarray(img3).convert("RGB").save("recon.png")

    mse = np.mean((img1 / 255. - img3 / 255.)**2)

    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr_res_valid_exp_yuv(img_ori_dir, img_up_dir, im_res_dir):
    global res_max
    global res_min

    img1 = Image.open(img_ori_dir).convert("RGB")
    img2 = Image.open(img_up_dir).convert("RGB")

    img1 = np.asarray(img1)
    img2 = np.asarray(img2)

    img_res = Image.open(im_res_dir).convert("RGB")
    img_res = np.asarray(img_res)

    img_res = img_res.astype(np.float32) * (res_max - res_min) / 255 + res_min

    img3 = rgb2yCbCr(img2 / 255. + img_res / 225.)

    img3 = yCbCr2rgb(img3) * 255
    img3 = img3.astype(np.uint8)

    Image.fromarray(img3).convert("RGB").save("recon.png")

    mse = np.mean((img1 / 255. - img3 / 255.)**2)

    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr_array_255(img1, img2):
    raise ValueError("psnr error")
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float("inf")
    else:
        return 20 * np.log10(255 / np.sqrt(mse))


def psnr_res_valid_exp_clipadd(img_ori_dir, img_up_dir, im_res_dir):
    global res_max
    global res_min

    img1 = Image.open(img_ori_dir).convert("RGB")
    img2 = Image.open(img_up_dir).convert("RGB")

    img1 = np.asarray(img1)
    img2 = np.asarray(img2)

    img_res = Image.open(im_res_dir).convert("RGB")
    img_res = np.asarray(img_res)

    img_res = img_res.astype(np.float32) * (res_max - res_min) / 255 + res_min

    img3 = img2.astype(np.float32) + img_res

    img3_max = np.max(img3)
    img3_min = np.min(img3)

    img3 = np.clip(img3, 0, 255)

    img3 = img3.astype(np.uint8)

    Image.fromarray(img3).convert("RGB").save("recon.png")

    mse = np.mean((img1 / 255. - img3 / 255.)**2)

    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def test_model():
    epoch_before = "001"
    num = 0
    psnr_mean = 0
    psnr_max, epoch_max = 0, 0
    dir = "checkpoints/mars_set_finetune_2022.10.31/web/images"
    for image in sorted(os.listdir(dir)):
        if len(re.findall("real", image)) != 0:
            dir_ori = os.path.join(dir, image)
        if len(re.findall("synthesized", image)) != 0:
            dir_contra = os.path.join(dir, image)
        if len(re.findall("epoch(\d+)_\d+_up_image", image)) != 0:
            epoch = re.findall("epoch(\d+)_\d+_up_image", image)[0]
            psnr_val = psnr_dir(dir_ori, dir_contra)
            if epoch == epoch_before:
                psnr_mean += psnr_val
                num += 1
            else:
                psnr_mean = psnr_mean / num
                print("epoch {0}, psnr is {1:.2f}".format(epoch, psnr_mean))
                if psnr_mean > psnr_max:
                    psnr_max = psnr_mean
                    epoch_max = epoch
                num = 0
                epoch_before = epoch
                psnr_mean = 0

    print("epoch max is {0}, psnr max is {1:.2f}".format(epoch_max, psnr_max))


def check_yuv():
    img1 = Image.open(
        "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_real_image.png"
    ).convert("RGB")
    img1 = np.asarray(img1).astype(np.float32) / 255

    img2 = np.clip(rgb2yCbCr(img1), 0, 1)
    img3 = np.clip(yCbCr2rgb(img2), 0, 255)

    img1_1 = (img1 * 255).astype(np.uint8)
    img3_1 = np.round(img3 * 255).astype(np.uint8)

    print("yuv recon psnr float is {}dB".format(
        psnr_array(img1 * 255, img3 * 255)))
    print("yuv recon psnr image is {}dB".format(psnr_array(img1_1, img3_1)))


def psnr_mars_retrain():
    base_dir = "vvc_results_retrain_2022_12_18/mars_retrain/test_200/images"
    real_files = []
    fake_files = []
    psnrs = []
    ssims = []
    ms_ssims = []
    for file in sorted(os.listdir(base_dir)):
        file_name = os.path.join(base_dir, file)
        if re.findall("real", file_name):
            real_files.append(file_name)
        if re.findall("synthesized", file_name):
            fake_files.append(file_name)

    for r, f in tqdm.tqdm(zip(real_files, fake_files), desc="test"):
        psnrs.append(psnr_dir(r, f))
        ssims.append(_calc_ssim_dir(r, f))
        ms_ssims.append(_calc_ms_ssim_dir(r, f))

    psnr_mean = np.mean(psnrs)
    ssim_mean = np.mean(ssims)
    ms_ssim_mean = np.mean(ms_ssims)
    print(f"psnr fake is {psnr_mean:.2f}\n ssim {ssim_mean: .4f}\n" \
          f"ms_ssim is {ms_ssim_mean: .4f}")


if __name__ == "__main__":
    ########################################### find best epoch #######################################
    # test_model()
    # check_yuv()
    ########################################### finish find best epoch ################################
   #  psnr_mars_retrain()
    dir_ori = input("ori dir: ")
    dir2 = input("contrast dir: ")
    print("PSNR is {0:.2f}dB".format(psnr_dir(dir_ori, dir2)))
    print("PSNR xjy is {0:.2f}dB".format(
        psnr_array_xjy(np.asarray(Image.open(dir_ori)),
                       np.asarray(Image.open(dir2)))))
    print("PSNR skimage is {0:.2f}dB".format(
        compare_psnr(np.asarray(Image.open(dir_ori)),
                     np.asarray(Image.open(dir2)))))
    print("SSIM is {0: .4f}".format(_calc_ssim_dir(dir_ori, dir2)))

############################################### test res uniform ##############################
# res_max = 0
# res_min = 0

# dir_ori = "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_real_image.png"
# dir2 = "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_synthesized_image.png"
# psnr_res(dir_ori, dir2)
# print("psnr is {0:.2f}dB".format(psnr_dir(dir_ori, dir2)))

# print("psnr max is {0:.2f}dB".format(psnr_res_valid_deal(
#    "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_real_image.png",
#    "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_synthesized_image.png",
#    "res.png")
# ))

# print("real vvc recon psnr is {0:.2f}dB".format(psnr_res_valid_deal(
#    "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_real_image.png",
#    "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_synthesized_image.png",
#    "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_res_recon.png"
# )))

# print("vvc exp1 psnr is {0:.2f}dB".format(psnr_res_valid_exp_clipadd(
#    "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_real_image.png",
#    "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_synthesized_image.png",
#    "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_res_recon.png"
# )))

##################################### finish test res uniform ##############################
