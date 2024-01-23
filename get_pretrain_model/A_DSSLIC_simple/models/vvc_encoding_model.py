# VVC encoder template

from genericpath import isdir
import os
from pickle import FALSE
import subprocess as subp
import threading
import re
import tqdm
from PIL import Image
import time
import shlex
from utility import psnr_dir
from test_bpp import calc_bpp_vvc_exp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from calc_ssim import _calc_ssim_dir


# comp qp now is 22
class EncodeWorker(threading.Thread):
    """
    used to encode the res image or the comp image
    """
    def __init__(self, img_path, base_path, qp, semaphore, 
                 is_comp=False):
        super(EncodeWorker, self).__init__()
        self.semaphore = semaphore
        self.base_path = "./train_temp/{}/".format(base_path)
        self.temp_path = "./train_temp/{}/temp/".format(base_path)
        self.bitstream_path = "./train_temp/{}/bit/".format(base_path)
        self.recon_path = "./train_temp/{}/recon/".format(base_path)
        self.img_path = img_path
        self.qp = qp
        self.is_comp = is_comp
        
        # self.vvc_encoder_path = "./EncoderAppStatic"
        # self.vvc_decoder_path = "./DecoderAppStatic"
        
        base_dir = "/mnt/disk10T/sxc/VVC_VTM/VVCSoftware_VTM-VTM-18.1/bin"
        self.vvc_encoder_path = "{}/EncoderAppStatic".format(base_dir)
        self.vvc_decoder_path = "{}/DecoderAppStatic".format(base_dir)
        
        self.ffmpeg_path = "/mnt/disk10T/sxc/ffmpeg-4.4-amd64-static/ffmpeg"
        self.recon_image_path = None
        self.bit_depth = 10
        self.pix_fmt = "yuv444p10le"
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
        self.is_comp = True
        comp_flag = True
        
        if not self.is_comp:
                if comp_flag:
                    self.lock.release()
                    self.semaphore.release()
                    return
        else:
            if comp_flag:
                self.qp = 22
                self.bitstream_path = self.base_path + '/bit_comp/'
                self.recon_path = self.base_path + "/comp/"
        
        stdout_vtm = open("{0}{1}vtm_log.txt".format(self.temp_path, img_name), 'a+')
        stdout_fmp = open("{0}{1}ffmpeg_log.txt".format(self.temp_path, img_name), 'a+')
        # 1200 x 864
        width = img.width
        height = img.height
        
        del(img)

        video_file_name = img_name + "video"
        
        if (os.path.exists("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name))): 
            os.remove("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name))
        
        
        # convert image to video
        print('\n# Convert image to video')
        convert_video_command = "{3} -i {0} -f rawvideo -pix_fmt {4} -dst_range 1 "\
            "{1}{2}_yuv.yuv".format(self.img_path, 
                                    self.temp_path, 
                                    video_file_name, 
                                    self.ffmpeg_path,
                                    self.pix_fmt
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
        print("Encoding")
        encode_command = "{0} -c ./vvc_cfg/encoder_intra_vtm.cfg -i "\
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
            
        subp.call(shlex.split(encode_command), 
            stdout=stdout_vtm, 
            shell=False
            )
        
        decode_file_name = img_name + "_dec"
        print('# Decoding')
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
        print('# Convert video to png image')
        
        convert_back_command = "{0} -f rawvideo -pix_fmt {7} "\
            "-s {1}x{2} -src_range 1 -i {3}{4}_rec.yuv -frames "\
            "1 -pix_fmt rgb24 {5}{6}.png".format(
                self.ffmpeg_path,
                width,
                height,
                self.temp_path,
                decode_file_name,
                self.recon_path,
                recon_file_name,
                self.pix_fmt
                )
        
        self.recon_image_path = "{0}{1}.png".format(self.recon_path, recon_file_name)
        
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
        if not isdir(dir):
            print("{0} is not exist, create dir: {0}".format(dir))
            os.mkdir(dir)
        else:
            print("{0} is exist".format(dir))
            if clear:
                print("--------------------clear directory content----------------------")
                for file in os.listdir(dir):
                    if not os.path.isdir(os.path.join(dir, file)):
                        os.remove(os.path.join(dir, file))


def run_train_encoder(img_path, path, qp, comp=True):
    semaphore = threading.BoundedSemaphore(1)
    t = EncodeWorker(img_path, path, qp, semaphore, is_comp=comp)
    t.set_properties(bit_depth=10, pix_fmt="yuv444p10le", chromafmt="444")
    t.check_file(clear=True)
    t.start()
    t.join()
    return t.recon_image_path
    

def run_vtm(image_path, path, qp, n_threads, bit_depth=10, pix_fmt="yuv444p10le", chromafmt="444", comp=False):
    """_summary_

    Args:
        image_path (str): the directory containing all the graph to be encoeded by vvc
        path (str): the directory name specified in encode_files to save the results
        qp (int): the quantize parameter, larger qp cause smaller bit-rate
        n_threads (int): total thread to encode the picture
        bit_depth (int): bit depth in vvc. Defaults to 10
        pix_fmt (str): pixel format for vvc to encode. Defaults to yuv44410lep
        chromafmt (str): correspoding to the pix_fmt. Defaults to 444
        comp (bool, optional): True to encode the comp_image. Defaults to False.
    """
    # comp and resid is waited to be done to control witch type of image to encode with vvc
    semaphore = threading.BoundedSemaphore(n_threads)
    threads_list = []
    
    file_list = create_file_list(image_path)
    
    for img_path in file_list:
        # img_path = img_path.encode('utf-8','backslashreplace').decode().replace("\\","/") 
        t = EncodeWorker(img_path, path, qp, semaphore, is_comp=comp)
        t.set_properties(bit_depth, pix_fmt, chromafmt)
        threads_list.append(t)
    
    t.show_properties()
    t.check_file(clear=True)
    
    for t in threads_list:
        t.start()
        print("{} threads is running".format(threading.active_count()))
        time.sleep(0.2)
    
    for t in threads_list:
        t.join()
    
    return t.bitstream_path, t.recon_path


def create_file_list(img_path):
    file_list = []
    for file in os.listdir(img_path):
        img_dir = os.path.join(img_path, file)
        if len(re.findall("res_image", file)) != 0:
            file_list.append(img_dir)
        if len(re.findall("comp_image", file)) != 0:
            file_list.append(img_dir)
    
    return file_list


###################################################################################
            ### code below are for the test of vvc encoding code
################################################################################
def test_run_vtm():
    # test run vtm function
    img_dir_path = "vvc_results/ADE20K_model/test_latest/images"
    base_path = "mars"
    Qp = 35
    start = time.time()
    run_vtm(img_dir_path, base_path, Qp, 20)

    tmp_path = "/mnt/disk10T/sxc/project/mars/comparison/DSSLIC/encode_files/mars/temp"
    
    for file in os.listdir(tmp_path):
        os.remove(os.path.join(tmp_path, file))
    
    print("time consume is {}".format(time.time() - start))


class ExpPSNRSaver:
    def __init__(self):
        self.psnr = {
            "yuv420p10le": [], 
            "yuv420p": [], 
            "yuv444p10le": [], 
            "yuv444p": []
        }
        self.bpp = {
            "yuv420p10le": [], 
            "yuv420p": [], 
            "yuv444p10le": [], 
            "yuv444p": []
        }
        
    def mean(self):
        psnr_new = {
            "yuv420p10le": [], 
            "yuv420p": [], 
            "yuv444p10le": [], 
            "yuv444p": []
        }
        for _type, _psnr_mean in self.psnr.items():
            psnr_new[_type] = "psnr_mean is {0:.2f}dB".format(np.mean(np.array(_psnr_mean)))
        
        bpp_new = {
            "yuv420p10le": [], 
            "yuv420p": [], 
            "yuv444p10le": [], 
            "yuv444p": []
        }
        for _type, _bpp_mean in self.bpp.items():
            bpp_new[_type] = "bpp_mean is {0:.4f}bpp".format(np.mean(np.array(_bpp_mean)))
        
        return psnr_new, bpp_new


class EncoderWorkerExp(EncodeWorker):
    def __init__(self, img_path, base_path, qp, semaphore, fp=None, is_comp=False):
        super(EncoderWorkerExp, self).__init__(img_path, base_path, qp, semaphore, is_comp)
        if fp:
            global saver_ade, saver_mars
            self.fp = fp
            self.saver_ade = saver_ade
            self.saver_mars = saver_mars 
        else:
            self.fp = None
        
    def run(self):
        # the function runs when the entry point start() method called
        # print("\ncurrent thread name: {}".format(thread_name))
        self.semaphore.acquire()
        self.lock.acquire()
        img = Image.open(self.img_path)
        
        img_name = self.img_path.split("/")[-1]
        img_name = img_name.split(".")[0]
        
        stdout_vtm = open("{0}{1}vtm_log.txt".format(self.temp_path, img_name), 'a+')
        stdout_fmp = open("{0}{1}ffmpeg_log.txt".format(self.temp_path, img_name), 'a+')
        # 1200 x 864
        width = img.width
        height = img.height
        
        del(img)
        # del(image)
        
        # print("img_name is : " + img_name)
        
        video_file_name = img_name + "video" + "_" + self.pix_fmt
        
        if (os.path.exists("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name))): 
            os.remove("{0}{1}_yuv.yuv".format(self.temp_path, video_file_name))
        
        
        # convert image to video
        print('\n# Convert image to video')
        convert_video_command = "{3} -i {0} -f rawvideo -pix_fmt {4} -dst_range 1 "\
            "{1}{2}_yuv.yuv".format(self.img_path, 
                                    self.temp_path, 
                                    video_file_name, 
                                    self.ffmpeg_path,
                                    self.pix_fmt
                                    )
        
        convert_video_command = shlex.split(convert_video_command)
            
        subp.call(convert_video_command, 
            shell=False, 
            stdout=stdout_fmp, 
            stderr=stdout_fmp
            )
        
        # os.remove(img_quan_path)a
        
        encode_file_name = img_name + "_enc" + "_" + self.pix_fmt
        # bit file
        # encoding the video
        print("Encoding")
        
        encode_command = "{0} -c ./vvc_cfg/encoder_intra_vtm.cfg -i "\
            "{1}{2}_yuv.yuv -b "\
            "{3}{4}.bin -q {7} "\
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
            
            
        subp.call(shlex.split(encode_command), 
            stdout=stdout_vtm, 
            shell=False
            )
        
        decode_file_name = img_name + "_dec" + self.pix_fmt
        print('# Decoding')
        decode_command = "{4} -b {0}{1}.bin "\
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
        
        recon_file_name = img_name + "_recon" + "_" + self.pix_fmt
        print('# Convert video to png image')
        
        convert_back_command = "{0} -f rawvideo -pix_fmt {7} "\
            "-s {1}x{2} -src_range 1 -i {3}{4}_rec.yuv -frames "\
            "1 -pix_fmt rgb24 {5}{6}.png".format(
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
        
        if self.fp:
            if len(re.findall("mars", self.img_path)) != 0:
                psnr_mars = psnr_dir(self.img_path, "{0}{1}.png".format(self.recon_path, recon_file_name))
                bpp_mars = calc_bpp_vvc_exp(
                        "{0}{1}.bin".format(self.bitstream_path, encode_file_name),
                        "mars"
                    )
                self.saver_mars.psnr[self.pix_fmt].append(psnr_mars)
                self.saver_mars.bpp[self.pix_fmt].append(bpp_mars)

                line_file = "mars" + img_name + "\t{2}\t\tpsnr is: {0:.2f}dB\tbpp is :{1:.4f}\n"\
                    .format(psnr_mars, bpp_mars, self.pix_fmt)
                self.fp.write(line_file)
            
            elif len(re.findall("ADE20K", self.img_path)) != 0:
                psnr_ade = psnr_dir(self.img_path, "{0}{1}.png".format(self.recon_path, recon_file_name))
                bpp_ade = calc_bpp_vvc_exp(
                        "{0}{1}.bin".format(self.bitstream_path, encode_file_name),
                        "ade20k"
                        )
                self.saver_ade.psnr[self.pix_fmt].append(psnr_ade)
                self.saver_ade.bpp[self.pix_fmt].append(bpp_ade)
                
                line_file = "ADE20K" + img_name  + "\t{2}\t\tpsnr is: {0:.2f}dB\tbpp is :{1:.4f}\n"\
                    .format(psnr_ade, bpp_ade, self.pix_fmt)
                self.fp.write(line_file)
        
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


class EncoderWorkerExpCalc(EncoderWorkerExp):
    def run(self):
        img_name = self.img_path.split("/")[-1]
        img_name = img_name.split(".")[0]
        encode_file_name = img_name + "_enc" + "_" + self.pix_fmt
        recon_file_name = img_name + "_recon" + "_" + self.pix_fmt
        if self.fp:
            if len(re.findall("mars", self.img_path)) != 0:
                psnr_mars = psnr_dir(self.img_path, "{0}{1}.png".format(self.recon_path, recon_file_name))
                bpp_mars = calc_bpp_vvc_exp(
                        "{0}{1}.bin".format(self.bitstream_path, encode_file_name),
                        "mars"
                    )
                self.saver_mars.psnr[self.pix_fmt].append(psnr_mars)
                self.saver_mars.bpp[self.pix_fmt].append(bpp_mars)

                line_file = "mars" + img_name + "\t{2}\t\tpsnr is: {0:.2f}dB\tbpp is :{1:.4f}\n"\
                    .format(psnr_mars, bpp_mars, self.pix_fmt)
                self.fp.write(line_file)
            
            elif len(re.findall("ADE20K", self.img_path)) != 0:
                psnr_ade = psnr_dir(self.img_path, "{0}{1}.png".format(self.recon_path, recon_file_name))
                bpp_ade = calc_bpp_vvc_exp(
                        "{0}{1}.bin".format(self.bitstream_path, encode_file_name),
                        "ade20k"
                        )
                self.saver_ade.psnr[self.pix_fmt].append(psnr_ade)
                self.saver_ade.bpp[self.pix_fmt].append(bpp_ade)
                
                line_file = "ADE20K" + img_name  + "\t{2}\t\tpsnr is: {0:.2f}dB\tbpp is :{1:.4f}\n"\
                    .format(psnr_ade, bpp_ade, self.pix_fmt)
                self.fp.write(line_file)
        return

    
def compress_file(img_path, bit_depth=10, pix_fmt="yuv444p10le", chromafmt="444", semaphore=None, exp_dir="1", fp=None, qp=None):
    if not qp:
        qp = 35
        
    flag = 0
    if semaphore is None:
        flag = 1
        semaphore = threading.BoundedSemaphore(1)
    if flag:
        t = EncoderWorkerExp(img_path, "exp{}".format(exp_dir), qp, semaphore)
    else:
        t = EncoderWorkerExp(img_path, "exp_{}_multi_QP{}".format(exp_dir, qp), qp, semaphore, fp)
        
    t.set_properties(bit_depth, pix_fmt, chromafmt)
    if flag:
        t.check_file(clear=True)
        t.start()
        t.join()
    
    return t


def calc_file(img_path, bit_depth, pix_fmt, chromafmt, semaphore=None, fp=None):
    qp = 35
    t = EncoderWorkerExpCalc(img_path, "contrast_exp_multi", qp, semaphore, fp) 
    t.set_properties(bit_depth, pix_fmt, chromafmt)

    return t


def test_avr_psnr_exp():
    semaphore = threading.BoundedSemaphore(40)
    test_dir = "results/ADE20K_model/test_latest/images"
    
    fp = open("vvc_coding_result_multi.txt", "w")
    
    thread_list = []
    """
    # for num, file in enumerate(sorted(os.listdir(test_dir))):
    #     test_path = os.path.join(test_dir, file)
    #     if len(re.findall("_(\d+_real_image)", file)) == 0:
    #         continue
    #     img_name = file.split(".")[0]
    #     print(img_name)

    #     t1 = compress_file(test_path, 10, "yuv420p10le", "420", semaphore, fp)
    #     t2 = compress_file(test_path, 8, "yuv420p", "420", semaphore, fp)
    #     t3 = compress_file(test_path, 10, "yuv444p10le", "444", semaphore, fp)
    #     t4 = compress_file(test_path, 8, "yuv444p", "444", semaphore, fp)
           
    #     thread_list.append([t1, t2, t3, t4])

    # t4.check_file(clear=True)
    
    test_dir = "datasets/mars/test"
    """
    for num, file in enumerate(sorted(os.listdir(test_dir))):
        if num >= 50:
            break
        test_path = os.path.join(test_dir, file)
        img_name = file.split(".")[0]

        t1 = compress_file(test_path, 10, "yuv420p10le", "420", semaphore, fp)
        t2 = compress_file(test_path, 8, "yuv420p", "420", semaphore, fp)
        t3 = compress_file(test_path, 10, "yuv444p10le", "444", semaphore, fp)
        t4 = compress_file(test_path, 8, "yuv444p", "444", semaphore, fp)
           
        thread_list.append([t1, t2, t3, t4])
    
    t4.check_file(clear=True)
    
    for t_sequential in thread_list:
        for t in t_sequential:
            t.start()
    
    for t_sequential in thread_list:
        for t in t_sequential:
            t.join()
    
    fp.close()
    

def test_avr_psnr_exp_1():
    test_dir = "results/ADE20K_model/test_latest/images"
    psnr_vvc_420p10 = 0
    bpp_420p10 = 0
    psnr_vvc_420p8 = 0
    bpp_420p8 = 0
    psnr_vvc_444p10 = 0
    bpp_444p10 = 0
    psnr_vvc_444p8 = 0
    bpp_444p8 = 0
    
    choice = "ade20k"
    
    fp = open("vvc_psnr.txt", "w")
    
    bit_depth = [10, 8, 10, 8]
    pix_fmts = ["yuv420p10le", "yuv420p", "yuv444p10le", "yuv444p"]
    chromafmts = ["420", "420", "444", "444"]
    
    for num, file in enumerate(os.listdir(test_dir)):
        test_path = os.path.join(test_dir, file)
        if len(re.findall("_(\d+_real_image)", file)) == 0:
            continue
        img_name = file.split(".")[0]

        for i in range(4):
            compress_file(test_path, bit_depth[i], pix_fmts[i], chromafmts[i])
            dir1 = "encode_files/contrast_exp/recon/{}_recon_{}.png".format(img_name, pix_fmts[i])
            dir2 = "encode_files/contrast_exp/bit/{}_enc_{}.bin".format(img_name, pix_fmts[i])
            
            if i == 0:
                psnr_vvc_420p10 +=  psnr_dir(test_path, dir1)
                bpp_420p10 += calc_bpp_vvc_exp(dir2, choice)
            if 1 == i:
                psnr_vvc_420p8 +=  psnr_dir(test_path, dir1)
                bpp_420p8 += calc_bpp_vvc_exp(dir2, choice)
            if 2 == i:
                psnr_vvc_444p10 +=  psnr_dir(test_path, dir1)
                bpp_444p10 += calc_bpp_vvc_exp(dir2, choice)
            if 3 == i:
                psnr_vvc_444p8 +=  psnr_dir(test_path, dir1)
                bpp_444p8 += calc_bpp_vvc_exp(dir2, choice)
    
    num = 50
    
    fp.writelines("-----------------------ADE20K begin---------------------------")
    fp.writelines("yuv420p10le: PSNR {0:.2f}dB, vvc_bpp {0:.4f}".
          format(psnr_vvc_420p10/num, bpp_420p10/num))
    fp.writelines("yuv420p8: PSNR {0:.2f}dB, vvc_bpp {0:.4f}".
          format(psnr_vvc_420p8/num, bpp_420p8/num))
    fp.writelines("yuv444p10le: PSNR {0:.2f}dB, vvc_bpp {0:.4f}".
          format(psnr_vvc_444p10/num, bpp_444p10/num))
    fp.writelines("yuv444p8: PSNR {0:.2f}dB, vvc_bpp {0:.4f}".
          format(psnr_vvc_444p8/num, bpp_444p8/num))
    fp.writelines("-----------------------ADE20K end---------------------------")
    
    choice = "mars"
    test_dir = "datasets/mars/test"
    psnr_vvc_420p10 = 0
    bpp_420p10 = 0
    psnr_vvc_420p8 = 0
    bpp_420p8 = 0
    psnr_vvc_444p10 = 0
    bpp_444p10 = 0
    psnr_vvc_444p8 = 0
    bpp_444p8 = 0
    
    for num, file in enumerate(os.listdir(test_dir)):
        if num >= 50:
            break
        test_path = os.path.join(test_dir, file)
        img_name = file.split(".")[0]
        
        for i in range(4):
            compress_file(test_path, bit_depth[i], pix_fmts[i], chromafmts[i])
            dir1 = "encode_files/contrast_exp/recon/{}_recon_{}.png".format(img_name, pix_fmts[i])
            dir2 = "encode_files/contrast_exp/bit/{}_enc_{}.bin".format(img_name, pix_fmts[i])
            
            if i == 0:
                psnr_vvc_420p10 +=  psnr_dir(test_path, dir1)
                bpp_420p10 += calc_bpp_vvc_exp(dir2, choice)
            if 1 == i:
                psnr_vvc_420p8 +=  psnr_dir(test_path, dir1)
                bpp_420p8 += calc_bpp_vvc_exp(dir2, choice)
            if 2 == i:
                psnr_vvc_444p10 +=  psnr_dir(test_path, dir1)
                bpp_444p10 += calc_bpp_vvc_exp(dir2, choice)
            if 3 == i:
                psnr_vvc_444p8 +=  psnr_dir(test_path, dir1)
                bpp_444p8 += calc_bpp_vvc_exp(dir2, choice)
    
    
    fp.writelines("-----------------------MARS begin---------------------------")
    fp.writelines("num" + str(num))
    
    num = 50
        
    fp.writelines("yuv420p10le: PSNR {0:.2f}dB, vvc_bpp {0:.4f}".
          format(psnr_vvc_420p10/num, bpp_420p10/num))
    fp.writelines("yuv420p8: PSNR {0:.2f}dB, vvc_bpp {0:.4f}".
          format(psnr_vvc_420p8/num, bpp_420p8/num))
    fp.writelines("yuv444p10le: PSNR {0:.2f}dB, vvc_bpp {0:.4f}".
          format(psnr_vvc_444p10/num, bpp_444p10/num))
    fp.writelines("yuv444p8: PSNR {0:.2f}dB, vvc_bpp {0:.4f}".
          format(psnr_vvc_444p8/num, bpp_444p8/num))
    fp.writelines("-----------------------MARS end---------------------------")
    
    fp.close()


def calc_metrics():
    semaphore = threading.BoundedSemaphore(1)
    test_dir = "results/ADE20K_model/test_latest/images"
    
    fp = open("vvc_coding_result_multi_calc_metrics.txt", "w")
    
    thread_list = []
    
    for num, file in enumerate(sorted(os.listdir(test_dir))):
        test_path = os.path.join(test_dir, file)
        if len(re.findall("_(\d+_real_image)", file)) == 0:
            continue
        img_name = file.split(".")[0]
        print(img_name)

        t1 = calc_file(test_path, 10, "yuv420p10le", "420", semaphore, fp)
        t2 = calc_file(test_path, 8, "yuv420p", "420", semaphore, fp)
        t3 = calc_file(test_path, 10, "yuv444p10le", "444", semaphore, fp)
        t4 = calc_file(test_path, 8, "yuv444p", "444", semaphore, fp)
           
        thread_list.append([t1, t2, t3, t4])

    # t4.check_file(clear=True)
    
    test_dir = "datasets/mars/test"
    
    for num, file in enumerate(sorted(os.listdir(test_dir))):
        if num >= 50:
            break
        test_path = os.path.join(test_dir, file)
        img_name = file.split(".")[0]

        t1 = calc_file(test_path, 10, "yuv420p10le", "420", semaphore, fp)
        t2 = calc_file(test_path, 8, "yuv420p", "420", semaphore, fp)
        t3 = calc_file(test_path, 10, "yuv444p10le", "444", semaphore, fp)
        t4 = calc_file(test_path, 8, "yuv444p", "444", semaphore, fp)
           
        thread_list.append([t1, t2, t3, t4])
    
    for t_sequential in thread_list:
        for t in t_sequential:
            t.start()
    
    for t_sequential in thread_list:
        for t in t_sequential:
            t.join()
    
    fp.close()


def test_base_line(qp_list):
    semaphore = threading.BoundedSemaphore(50)
    bpp_means = np.array([])
    psnr_means = np.array([])
    for qp in qp_list:
        thread_list = []
        num = 0
        ori_dirs = []
        test_dir = "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images"
        for file in sorted(os.listdir(test_dir)):
            if len(re.findall("real_image", file)) != 0:   
                test_path = os.path.join(test_dir, file)
                ori_dirs.append(test_path)
                img_name = file.split(".")[0]
                print(img_name)
                t = compress_file(test_path, 10, "yuv444p10le", "444", semaphore,
                                  exp_dir="baseline", qp=qp)
                
                thread_list.append(t)
        
        t.check_file(clear=True)
    
        for t in thread_list:
            num += 1  
            t.start()
            print("{} threads is running".format(threading.active_count()))
            time.sleep(0.2)
        
        
        for t in thread_list:
            t.join()
        
        bit_path = t.bitstream_path
        ori_path = test_dir
        recon_path = t.recon_path
        bpp = 0
        psnr = 0
        for file in sorted(os.listdir(bit_path)):
            bpp += calc_bpp_vvc_exp(
                os.path.join(bit_path, file),
                "mars_test"
                )

        for im1, im2 in zip(ori_dirs, sorted(os.listdir(recon_path))):
            psnr += psnr_dir(
                im1,
                os.path.join(recon_path, im2)
            )
            
        bpp_means = np.append(bpp_means, bpp/num)
        psnr_means = np.append(psnr_means, psnr/num)
    
    print("----------------------bpp---------------------------")
    print(bpp_means)
    print("----------------------psnr---------------------------")
    print(psnr_means)
    print("----------------------end---------------------------")
    plt.plot(bpp_means, psnr_means)
    f = plt.gcf()
    f.savefig("baseline.png")
    f.clear()


def test_base_line_data(qp_list):
    semaphore = threading.BoundedSemaphore(50)
    bpp_means = np.array([])
    psnr_means = np.array([])
    ssim_means = np.array([])
    for qp in qp_list:
        thread_list = []
        num = 0
        ori_dirs = []
        test_dir = "vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images"
        for file in sorted(os.listdir(test_dir)):
            if len(re.findall("real_image", file)) != 0:   
                test_path = os.path.join(test_dir, file)
                ori_dirs.append(test_path)
                img_name = file.split(".")[0]
                print(img_name)
                t = compress_file(test_path, 10, "yuv444p10le", "444", semaphore,
                                  exp_dir="baseline", qp=qp)
                
                thread_list.append(t)
        
        print(qp)
        t.check_file(clear=True)
    
        for t in thread_list:
            num += 1  
            t.start()
            print("{} threads for vvc is running".format(threading.active_count() - 1))
            time.sleep(0.2)
        
        
        for t in thread_list:
            t.join()
        
        bit_path = t.bitstream_path
        ori_path = test_dir
        recon_path = t.recon_path
        bpp = 0
        psnr = 0
        ssim = 0
        for file in sorted(os.listdir(bit_path)):
            bpp += calc_bpp_vvc_exp(
                os.path.join(bit_path, file),
                "mars_test"
                )

        for im1, im2 in zip(ori_dirs, sorted(os.listdir(recon_path))):
            psnr += psnr_dir(
                im1,
                os.path.join(recon_path, im2)
            )
            ssim += _calc_ssim_dir(
                im1,
                os.path.join(recon_path, im2)
            )
            
        bpp_means = np.append(bpp_means, bpp/num)
        psnr_means = np.append(psnr_means, psnr/num)
        ssim_means = np.append(ssim_means, ssim/num)
    
    print("----------------------bpp---------------------------")
    print(bpp_means)
    print("----------------------psnr---------------------------")
    print(psnr_means)
    print("----------------------end---------------------------")
    plt.plot(bpp_means, psnr_means)
    f = plt.gcf()
    f.savefig("baseline.png")
    f.clear()
    
    qp_list_pd = pd.Series(qp_list).to_frame(name='qp')
    bpp_means_pd = pd.Series(bpp_means).to_frame(name='mean_bpp')
    psnr_means_pd = pd.Series(psnr_means).to_frame(name='mean_PSNR')
    ssim_means_pd = pd.Series(ssim_means).to_frame(name='mean_ssim')

    data_df = pd.concat([qp_list_pd, bpp_means_pd, psnr_means_pd, ssim_means_pd], axis=1)

    data_df.to_excel("data_baseline.xlsx")


def calc_metrics(qp_list):
    pass


if __name__ == "__main__":
    # raise ValueError("do not run it")
############################################### compress the files ##############################
    # saver_ade = ExpPSNRSaver()
    # saver_mars = ExpPSNRSaver()
    # test_avr_psnr_exp()
    
    # psnr_mars, bpp_mars = saver_mars.mean()
    # psnr_ade, bpp_ade = saver_ade.mean()
    
    # print("------------------ade20k--------------------")
    # print(psnr_ade)
    # print(bpp_ade)
    # print("------------------mars--------------------")
    # print(psnr_mars)
    # print(bpp_mars)
################################################################# finish compress the file ##############################
######################################### evaluation below ############################################################
    # saver_ade = ExpPSNRSaver()
    # saver_mars = ExpPSNRSaver()
    # calc_metrics()
    
    # psnr_mars, bpp_mars = saver_mars.mean()
    # psnr_ade, bpp_ade = saver_ade.mean()
    
    # print("------------------ade20k--------------------")
    # print(psnr_ade)
    # print(bpp_ade)
    # print("------------------mars--------------------")
    # print(psnr_mars)
    # print(bpp_mars)
############################################### finish evaluation ##############################

########################################  the curve os R-D for vvc  ###############################################
    qp_lists = [i for i in range(12, 48)]
    
    test_base_line_data(qp_lists)
    
########################################  finish the curve os R-D for vvc  ############################################### 
########################################  compress a file  ############################################### 
    # compress_file("vvc_mars_results/mars_set_finetune_2022.10.31/test_120/images/3000_real_image.png")
########################################  finish compress a file  ############################################### 


   