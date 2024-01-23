import os
import cv2


def filesize(filepath):
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError("Invalid file {0}.".format(filepath))
    file_stat =  os.stat(filepath)
    return file_stat.st_size


def calc_bpp_data(data, width, height):
    raise NotImplementedError()


def calc_bpp_all(comp_path, res_vvc_path, height, width):
    bpp_comp = filesize(comp_path) * 8.0 / (height * width)
    bpp_res = filesize(res_vvc_path) * 8.0 / (height * width)
    # print('bit: %d' %a)
    # print('pixel: %d' %(height * width))
    # print('bpp_total: {0:%7.5f}, comp_bpp: {1}, vvc_bpp: {2}'.format(
    #     bpp_comp + bpp_res, 
    #     bpp_comp, 
    #     bpp_res
    #     )
    #       )
    return bpp_comp + bpp_res, bpp_res


def calc_bpp_vvc_exp(file, choice="mars"):
    # only used for the contrast experiment
    if choice == "mars":
        height = 1152
        width = 1600
    else:
        raise ValueError("no setting")
    
    bpp = filesize(file) * 8.0 / (height * width)
    # print("vvc bpp is {}".format(bpp))
    return bpp
    

if __name__ == "__main__":

    out_filepath = "encode_files/exp_valid_multi_QP49/bit/0_enc_yuv444p.bin"
    out_filepath = input("test_path: ")

    height = 1152
    width = 1600
    a = filesize(out_filepath) * 8.0
    bpp = filesize(out_filepath) * 8.0 / (height * width)
    bpp2 = 0
    print('bit: %d' %a)
    print('pixel: %d' %(height * width))
    print('bpp: %7.4f' %bpp)

    
    