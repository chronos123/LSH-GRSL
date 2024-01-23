import os
import threading
import re
import shlex
import cv2
import subprocess as subp
from PIL import Image
import time
from argparse import ArgumentParser
import tqdm


class JPEGWorker(threading.Thread):
    def __init__(self, img_path, base_path, quality, semaphore: threading.BoundedSemaphore, mode="openjpeg", timer=None):
        super().__init__()
        self.lock = threading.RLock()
        self.mode = mode
        self.timer = timer
        self.semaphore = semaphore
        self.img_path = img_path
        self.base_path = base_path
        self.ffmpeg = "static_file/ffmpeg"
        
        self.log_dir = os.path.join(base_path, "log")
        self.compress_dir = os.path.join(base_path, "compress")
        self.recon_dir = os.path.join(base_path, "recon")
        self.codec = "jpeg2000"
        os.environ['LD_LIBRARY_PATH'] = "static_file/jpeg2000"
        self.openjpeg_enc = "static_file/jpeg2000/opj_compress"
        self.openjpeg_dec = "static_file/jpeg2000/opj_decompress"
        if self.codec == "jpeg":
            assert 0 < quality <= 1000
        # jpeg2000, mjpeg, jpeg
        self.qf = quality
        assert 2 <= self.qf <= 200
        # 2-31
    
    def check_dir(self, clear=False):
        dirs = [self.base_path ,self.log_dir, self.compress_dir, self.recon_dir]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
            else:
                print(f"{dir} exists")
                if clear:
                    for file in os.listdir(dir):
                        if not os.path.isdir(os.path.join(dir, file)):
                            os.remove(os.path.join(dir, file))
                    print("-----------------clear dir------------------")
    
    def run(self):
        self.semaphore.acquire()
        self.lock.acquire()
        img_name = self.img_path.split("/")[-1]
        img_name = img_name.split(".")[0]
        log_dir = os.path.join(self.log_dir, f"{img_name}-{self.mode}-log.txt")
        logfile_fmp = open(log_dir, "a+")
        if self.mode == "ffmpeg":
            if self.codec == "jpeg":
                assert 0 < self.qf <= 1000
                img = cv2.imread(self.img_path)
                enc_start = time.time()
                cv2.imwrite(
                    f"{self.compress_dir}/{img_name}_compress.jpg",
                    img,
                    [cv2.IMWRITE_JPEG_QUALITY, self.qf]
                    )
                enc_time = time.time() - enc_start
                
            else:
                compress_cmd = [
                    self.ffmpeg,
                    '-i',
                    self.img_path,
                    '-c:v',
                    self.codec,
                    '-q:v',
                    str(self.qf),
                    f"{self.compress_dir}/{img_name}_compress.jpg"
                ]
                
                # /mnt/disk10T/sxc/ffmpeg-4.4-amd64-static/ffmpeg -i datasets/NASA-final/test/0.png -c:v jpeg2000 -q:v 15 test_codes/compressed_jpeg2000/jpeg_ffmpeg/recon/0_recon.jpg
                # 目前下面的指令可以成功压缩图片
                # /mnt/disk10T/sxc/ffmpeg-4.4-amd64-static/ffmpeg -i datasets/NASA-final/test/0.png -q 20 test_codes/compressed_jpeg2000/jpeg_ffmpeg/recon/0_recon.jpg
                
                subp.run(compress_cmd, shell=False, stdout=logfile_fmp, stderr=logfile_fmp)
            
            recon_cmd = [
                self.ffmpeg,
                "-i",
                f"{self.compress_dir}/{img_name}_compress.jpg",
                f"{self.recon_dir}/{img_name}_recon.png"
            ]
            dec_start = time.time()
            subp.run(recon_cmd, shell=False, stdout=logfile_fmp, stderr=logfile_fmp)
            dec_time = time.time() - dec_start
            self.timer.update(enc_time, dec_time)
            self.semaphore.release()
            self.lock.release()
        
        else:
            # evaluation code/openjpeg/build/bin/opj_compress \
            # -i src -o dst.jp2 -r ratio(压缩几倍)   
            
            # evaluation code/openjpeg/build/bin/opj_decompress \
            # -i src.jp2 -o dst 
            compress_cmd = [
                    self.openjpeg_enc,
                    '-i',
                    self.img_path,
                    '-o',
                    f"{self.compress_dir}/{img_name}_compress.jp2",
                    "-r",
                    str(self.qf),
                ]
            enc_st = time.time()
            subp.run(compress_cmd, shell=False, stdout=logfile_fmp, stderr=logfile_fmp)
            enc_time = time.time() - enc_st
            
            recon_cmd = [
                self.openjpeg_dec,
                "-i",
                f"{self.compress_dir}/{img_name}_compress.jp2",
                "-o",
                f"{self.recon_dir}/{img_name}_recon.png",
            ]
            dec_st = time.time()
            subp.run(recon_cmd, shell=False, stdout=logfile_fmp, stderr=logfile_fmp)
            dec_time = time.time() - dec_st
            if self.timer:
                self.timer.update(enc_time, dec_time)
            self.semaphore.release()
            self.lock.release()    


def run_jpeg2000():
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--n_threads",
        type=int,
        default=20,
        help="how many threads running at a time"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="the data to compress"
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=20,
        help="the data to compress"
    )
    
    args = parser.parse_args()
    
    semaphore = threading.BoundedSemaphore(args.n_threads)
    threads_list = []
    
    for file in sorted(os.listdir(args.dataset)):
        img_path = os.path.join(args.dataset, file)
        t = JPEGWorker(img_path, f"encode_files_jpeg2000/jpeg2000_{args.quality}", args.quality, semaphore)
        threads_list.append(t)
        
    t.check_dir(clear=True)
    
    pbar = tqdm.tqdm(total=len(threads_list))
    pbar.set_description("jpeg2000 encoding")
    
    threads_list_1 = []
    threads_list.reverse()
    while threads_list:
        if threading.active_count() < args.n_threads + 1:
            t = threads_list.pop()
            threads_list_1.append(t)
            t.start()
            # print("{} threads is running".format(threading.active_count()))
            time.sleep(0.2)
            pbar.update(1)
    
    for t in threads_list_1:
        t.join()  


if __name__ == "__main__":
    run_jpeg2000()
    