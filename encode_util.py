import os
import threading
import re
import shlex
import cv2
import subprocess as subp
from PIL import Image
from tqdm import tqdm
import time
import math
import numpy as np
import pandas as pd
from argparse import ArgumentParser


class VVCEncodeWorker(threading.Thread):
    """
    used to encode the res image or the comp image
example:
    t1 = EncodeWorker(
        "datasets/NASA-final/val/119.png",
        "test_vvc_encodes",
        50,
        semaphore,
        encode_save = False
    )
    """
    def __init__(self, img_path, base_path, qp, semaphore, 
                 is_comp=False, encode_save=True):
        super(VVCEncodeWorker, self).__init__()
        self.semaphore = semaphore
        if encode_save:
            self.base_path = "encode_files/{}/".format(base_path)
        else:
            self.base_path = base_path + '/'
            
        self.temp_path = self.base_path + "temp/"
        self.bitstream_path = self.base_path + "bit/"
        self.recon_path = self.base_path + "recon/"
        self.img_path = img_path
        self.qp = qp
        self.is_comp = is_comp
        
        # self.vvc_encoder_path = "./EncoderAppStatic"
        # self.vvc_decoder_path = "./DecoderAppStatic"
        
        base_dir = "static_file"
        self.vvc_encoder_path = "{}/EncoderAppStatic".format(base_dir)
        self.vvc_decoder_path = "{}/DecoderAppStatic".format(base_dir)
        
        self.ffmpeg_path = "static_file/ffmpeg"
        
        self.bit_depth = 8
        self.pix_fmt = "yuv444p"
        self.chromafmt = "444"
        self.lock = threading.RLock()
        
    
    def set_properties(self, bit_depth, pix_fmt, chromafmt):
        self.bit_depth = bit_depth
        self.pix_fmt = pix_fmt
        self.chromafmt = chromafmt
    
    def show_properties(self):
        print("bit depth {}, pixel format {} chroma format {} is used"
              .format(self.bit_depth, self.pix_fmt, self.chromafmt))
    
    def check_file(self, clear=False):
        self.checkdir(self.base_path, clear)
        self.checkdir(self.temp_path, clear)
        self.checkdir(self.bitstream_path, clear)
        self.checkdir(self.recon_path, clear)
        self.checkdir(self.base_path + "comp/", clear)
        self.checkdir(self.base_path + "bit_comp/", clear)
    
    def run(self):
        # the function runs when the entry point start() method called
        # thread_name = threading.current_thread().name
        # print("\ncurrent thread name: {}".format(thread_name))
        self.semaphore.acquire()
        self.lock.acquire()
        comp_flag = False
        img = Image.open(self.img_path)
        
        img_name = self.img_path.split("/")[-1]
        img_name = img_name.split(".")[0]
        
        try:
            content = re.findall("(\d+_.*?_image)", img_name)
            
            img_name = content[0].strip("0")
        except:
            img_name = img_name
        
        if len(re.findall("comp_image", img_name)) != 0:
            comp_flag = True
        
        if not self.is_comp:
                if comp_flag:
                    self.lock.release()
                    self.semaphore.release()
                    return
        else:
            if comp_flag:
                self.qp = 22
                self.bitstream_path = self.base_path + 'bit_comp/'
                self.recon_path = self.base_path + "comp/"
        
        stdout_vtm = open("{0}{1}vtm_log.txt".format(self.temp_path, img_name), 'a+')
        stdout_fmp = open("{0}{1}ffmpeg_log.txt".format(self.temp_path, img_name), 'a+')
        width = img.width
        height = img.height
        
        del(img)

        video_file_name = img_name + "video"
        
        if (os.path.exists("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name))): 
            os.remove("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name))
        
        
        # convert image to video
        # print('\n# Convert image to video')
        convert_video_command = "{3} -i {0} -s {5}x{6} -pix_fmt {4} "\
            "{1}{2}_yuv.yuv".format(self.img_path, 
                                    self.temp_path, 
                                    video_file_name, 
                                    self.ffmpeg_path,
                                    self.pix_fmt,
                                    width,
                                    height
                                    )
        
        convert_video_command = shlex.split(convert_video_command)
            
        subp.call(convert_video_command, 
            shell=False, 
            stdout=stdout_fmp, 
            stderr=stdout_fmp
            )
        
        encode_file_name = img_name + "_enc"
        # bit file
        # encoding the video
        
        # print("Encoding")
        encode_command = "{0} -c /home/sxc/DSSLIC/vvc_cfg/encoder_intra_vtm.cfg -i "\
            "{1}{2}_yuv.yuv -b "\
            "{3}{4}.vvc -q {7} "\
            "-wdt {5} -hgt {6} -f 1 -fr 1 --InputChromaFormat={9} "\
            "--InputBitDepth={8} --OutputBitDepth={8} --InternalBitDepth={8}"\
            .format(self.vvc_encoder_path, 
                    self.temp_path, 
                    video_file_name,
                    self.bitstream_path,
                    encode_file_name,
                    width,
                    height,
                    self.qp,
                    self.bit_depth,
                    self.chromafmt
                    )
        # --InputBitDepth={8} --OutputBitDepth={8} 
        
        subp.call(shlex.split(encode_command), 
            stdout=stdout_vtm, 
            shell=False
            )
        
        decode_file_name = img_name + "_dec"
        # print('# Decoding')
        decode_command = "{4} -b {0}{1}.vvc "\
            "-o {2}{3}_rec.yuv".format(
                self.bitstream_path,
                encode_file_name,
                self.temp_path,
                decode_file_name,
                self.vvc_decoder_path
                )
            
        subp.call(shlex.split(decode_command), 
            stdout=stdout_vtm, 
            shell=False)
        
        recon_file_name = img_name + "_recon"
        # print('# Convert video to png image')
        
        convert_back_command = "{0} -pix_fmt {7} "\
            "-s {1}x{2} -i {3}{4}_rec.yuv -vframes "\
            "1 {5}{6}.png".format(
                self.ffmpeg_path,
                width,
                height,
                self.temp_path,
                decode_file_name,
                self.recon_path,
                recon_file_name,
                self.pix_fmt
                )
        
        if (os.path.exists("{0}{1}_rec.png".format(self.recon_path, recon_file_name))): 
            os.remove("{0}{1}_rec.png".format(self.recon_path, recon_file_name))
        
        subp.call(shlex.split(convert_back_command), 
            shell=False, 
            stdout=stdout_fmp, 
            stderr=stdout_fmp)
        
        if os.path.exists("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name)):
            try:
                os.remove("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name))
            except OSError:
                raise Exception("can not delete temp file")
            
        if os.path.exists("{0}{1}_rec.yuv".format(self.temp_path, decode_file_name)):
            try:
                os.remove("{0}{1}_rec.yuv".format(self.temp_path, decode_file_name))
            except OSError:
                raise Exception("can not delete temp file")
        
        self.lock.release()
        self.semaphore.release()
        
        return
    
    def checkdir(self, dir, clear=False):
        if not os.path.isdir(dir):
            print("{0} is not exist, create dir: {0}".format(dir))
            os.mkdir(dir)
            return False
        else:
            print("{0} is exist".format(dir))
            if clear:
                print("--------------------clear directory content----------------------")
                for file in os.listdir(dir):
                    if not os.path.isdir(os.path.join(dir, file)):
                        os.remove(os.path.join(dir, file))
            else:
                return True


class VVCEncodeWorkerMars(threading.Thread):
    """
    used to encode the res image or the comp image
example:
    t1 = EncodeWorker(
        "datasets/NASA-final/val/119.png",
        "test_vvc_encodes",
        50,
        semaphore,
        encode_save = False
    )
    """
    def __init__(self, img_path, base_path, qp, semaphore, 
                 is_comp=False, encode_save=True, args=None):
        super().__init__()
        self.semaphore = semaphore
        if encode_save:
            if args is not None:
                self.base_path = f"encode_files_{args.output}/{base_path}/"
            else:
                self.base_path = "encode_files_mars_new_set/{}/".format(base_path)
        else:
            self.base_path = base_path + '/'
            
        self.temp_path = self.base_path + "temp/"
        self.bitstream_path = self.base_path + "bit/"
        self.recon_path = self.base_path + "recon/"
        self.img_path = img_path
        self.qp = qp
        self.is_comp = is_comp
        
        # self.vvc_encoder_path = "./EncoderAppStatic"
        # self.vvc_decoder_path = "./DecoderAppStatic"
        
        base_dir = "static_file"
        self.vvc_encoder_path = "{}/EncoderAppStatic".format(base_dir)
        self.vvc_decoder_path = "{}/DecoderAppStatic".format(base_dir)
        
        self.ffmpeg_path = "static_file/ffmpeg"
        
        self.bit_depth = 8
        self.pix_fmt = "yuv444p"
        self.chromafmt = "444"
        self.lock = threading.RLock()
        
    
    def set_properties(self, bit_depth, pix_fmt, chromafmt):
        self.bit_depth = bit_depth
        self.pix_fmt = pix_fmt
        self.chromafmt = chromafmt
    
    def show_properties(self):
        print("bit depth {}, pixel format {} chroma format {} is used"
              .format(self.bit_depth, self.pix_fmt, self.chromafmt))
    
    def check_file(self, clear=False):
        self.checkdir(self.base_path, clear)
        self.checkdir(self.temp_path, clear)
        self.checkdir(self.bitstream_path, clear)
        self.checkdir(self.recon_path, clear)
        self.checkdir(self.base_path + "comp/", clear)
        self.checkdir(self.base_path + "bit_comp/", clear)
    
    def run(self):
        # the function runs when the entry point start() method called
        # thread_name = threading.current_thread().name
        # print("\ncurrent thread name: {}".format(thread_name))
        self.semaphore.acquire()
        self.lock.acquire()
        comp_flag = False
        img = Image.open(self.img_path)
        
        img_name = self.img_path.split("/")[-1]
        img_name = img_name.split(".")[0]
        
        try:
            content = re.findall("(\d+_.*?_image)", img_name)
            
            img_name = content[0].strip("0")
        except:
            img_name = img_name
        
        if len(re.findall("comp_image", img_name)) != 0:
            comp_flag = True
        
        if not self.is_comp:
                if comp_flag:
                    self.lock.release()
                    self.semaphore.release()
                    return
        else:
            if comp_flag:
                self.qp = 22
                self.bitstream_path = self.base_path + 'bit_comp/'
                self.recon_path = self.base_path + "comp/"
        
        stdout_vtm = open("{0}{1}vtm_log.txt".format(self.temp_path, img_name), 'a+')
        stdout_fmp = open("{0}{1}ffmpeg_log.txt".format(self.temp_path, img_name), 'a+')
        width = img.width
        height = img.height
        
        del(img)

        video_file_name = img_name + "video"
        
        if (os.path.exists("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name))): 
            os.remove("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name))
        
        
        # convert image to video
        # print('\n# Convert image to video')
        convert_video_command = "{3} -i {0} -s {5}x{6} -pix_fmt {4} "\
            "{1}{2}_yuv.yuv".format(self.img_path, 
                                    self.temp_path, 
                                    video_file_name, 
                                    self.ffmpeg_path,
                                    self.pix_fmt,
                                    width,
                                    height
                                    )
        
        convert_video_command = shlex.split(convert_video_command)
            
        subp.call(convert_video_command, 
            shell=False, 
            stdout=stdout_fmp, 
            stderr=stdout_fmp
            )
        
        encode_file_name = img_name + "_enc"
        # bit file
        # encoding the video
        
        # print("Encoding")
        encode_command = "{0} -c static_file/encoder_intra_vtm.cfg -i "\
            "{1}{2}_yuv.yuv -b "\
            "{3}{4}.vvc -q {7} "\
            "-wdt {5} -hgt {6} -f 1 -fr 1 --InputChromaFormat={9} "\
            "--InputBitDepth={8} --OutputBitDepth={8} --InternalBitDepth={8}"\
            .format(self.vvc_encoder_path, 
                    self.temp_path, 
                    video_file_name,
                    self.bitstream_path,
                    encode_file_name,
                    width,
                    height,
                    self.qp,
                    self.bit_depth,
                    self.chromafmt
                    )
        # --InputBitDepth={8} --OutputBitDepth={8} 
        
        subp.call(shlex.split(encode_command), 
            stdout=stdout_vtm, 
            shell=False
            )
        
        decode_file_name = img_name + "_dec"
        # print('# Decoding')
        decode_command = "{4} -b {0}{1}.vvc "\
            "-o {2}{3}_rec.yuv".format(
                self.bitstream_path,
                encode_file_name,
                self.temp_path,
                decode_file_name,
                self.vvc_decoder_path
                )
            
        subp.call(shlex.split(decode_command), 
            stdout=stdout_vtm, 
            shell=False)
        
        recon_file_name = img_name + "_recon"
        # print('# Convert video to png image')
        
        convert_back_command = "{0} -pix_fmt {7} "\
            "-s {1}x{2} -i {3}{4}_rec.yuv -vframes "\
            "1 {5}{6}.png".format(
                self.ffmpeg_path,
                width,
                height,
                self.temp_path,
                decode_file_name,
                self.recon_path,
                recon_file_name,
                self.pix_fmt
                )
        
        if (os.path.exists("{0}{1}_rec.png".format(self.recon_path, recon_file_name))): 
            os.remove("{0}{1}_rec.png".format(self.recon_path, recon_file_name))
        
        subp.call(shlex.split(convert_back_command), 
            shell=False, 
            stdout=stdout_fmp, 
            stderr=stdout_fmp)
        
        if os.path.exists("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name)):
            try:
                os.remove("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name))
            except OSError:
                raise Exception("can not delete temp file")
            
        if os.path.exists("{0}{1}_rec.yuv".format(self.temp_path, decode_file_name)):
            try:
                os.remove("{0}{1}_rec.yuv".format(self.temp_path, decode_file_name))
            except OSError:
                raise Exception("can not delete temp file")
        
        self.lock.release()
        self.semaphore.release()
        
        return
    
    def checkdir(self, dir, clear=False):
        if not os.path.isdir(dir):
            print("{0} is not exist, create dir: {0}".format(dir))
            os.makedirs(dir)
            return False
        else:
            print("{0} is exist".format(dir))
            if clear:
                print("--------------------clear directory content----------------------")
                for file in os.listdir(dir):
                    if not os.path.isdir(os.path.join(dir, file)):
                        os.remove(os.path.join(dir, file))
            else:
                return True


class MetricAvg:
    def __init__(self, ms_ssim=False):
        self.count = 0
        self.bpp = 0 
        self.psnr = 0
        if ms_ssim:
            self.ms_ssim = 0

    def update(self, bpp, psnr, ms_ssim=None):
        if ms_ssim:
            assert ms_ssim is not None
            self.ms_ssim += ms_ssim
        self.bpp += bpp
        self.psnr += psnr
        self.count += 1

    def get_metric(self, ms_ssim=None):
        if ms_ssim:
            return self.bpp / self.count, self.psnr / self.count, self.ms_ssim / self.count
        else:
            return self.bpp / self.count, self.psnr / self.count
 

def filesize(filepath):
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError("Invalid file {0}.".format(filepath))
    file_stat =  os.stat(filepath)
    return file_stat.st_size


def calc_bpp_conventional_exp(file, choice="mars"):
    # only used for the contrast experiment
    if choice == "mars":
        height = 1152
        width = 1600
    else:
        raise ValueError("no settings")
    
    bpp = filesize(file) * 8.0 / (height * width)
    # print("vvc bpp is {}".format(bpp))
    return bpp


def psnr_dir(img1_dir, img2_dir):
   img1 = Image.open(img1_dir).convert("RGB")
   img1 = np.asarray(img1)
   img2 = Image.open(img2_dir).convert("RGB")
   img2 = np.asarray(img2)
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_metrics(qp, args):
    avg = MetricAvg()
    base_path = f"encode_files_{args.output}/{qp}"
    bit_path = os.path.join(base_path, "bit")
    rec_path = os.path.join(base_path, "recon")
    ori_path = "/mnt/disk4T/sxc/mars/dataset/new_set"
    
    for ori_img in tqdm.tqdm(sorted(os.listdir(ori_path)), desc="calc_metrics"):
        image_name = ori_img.split(".")[0]
        ori_img_path = os.path.join(ori_path, ori_img)
        rec_img_path = os.path.join(rec_path, f"{image_name}_recon.png")
        bpp_path = os.path.join(bit_path, f"{image_name}_enc.vvc")

        _psnr = psnr_dir(ori_img_path, rec_img_path)
        _bpp = calc_bpp_conventional_exp(bpp_path, choice="mars_new")
        avg.update(_bpp, _psnr)
        
    avg_bpp , avg_psnr = avg.get_metric()
    
    return {
        "qp": int(qp),
        "bpp": f"{avg_bpp:.4f}",
        "psnr": f"{avg_psnr:.2f}",
    }


def calc_all(args):
    for qp in [28]:
        result = calc_metrics(qp, args)
        result = pd.Series(result).to_frame().T
        print(result)
        results = results.append(result)
    
    # results = results.applymap(lambda x: f'{x:.2f}')
    results.to_excel(f"vvc_new_{args.output}.xlsx")


def vvc_encoding(qp, args):
    base_path = args.dataset
    threads_list = []
    semaphore = threading.BoundedSemaphore(56)
    for file in tqdm(os.listdir(base_path)):
        img_path = os.path.join(base_path, file)
        vvc_encoder = VVCEncodeWorkerMars(img_path, base_path=f"{qp}", qp=qp, semaphore=semaphore, args=args)
        vvc_encoder.setDaemon(True)
        threads_list.append(vvc_encoder)
    vvc_encoder.check_file(clear=True)
    pbar = tqdm(total=len(threads_list))
    pbar.set_description("vvc encoding")
    
    threads_list_1 = []
    threads_list.reverse()
    while threads_list:
        if threading.active_count() < 56 + 1:
            t = threads_list.pop()
            threads_list_1.append(t)
            t.start()
            # print("{} threads is running".format(threading.active_count()))
            time.sleep(0.2)
            pbar.update(1)
    
    for t in threads_list_1:
        t.join()  


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True
    )
    
    args = parser.parse_args()
    
    for qp in [28]:
        vvc_encoding(qp, args)

    calc_all(args)
