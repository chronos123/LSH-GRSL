import os
import subprocess as subp
import threading
import tqdm
import time
from argparse import ArgumentParser
import shlex
import re
from PIL import Image


class BpgEncodeWorker(threading.Thread):
    def __init__(self, img_path, base_path, qp, semaphore, 
                 is_comp=False, encode_save=True, name=None):
        super(BpgEncodeWorker, self).__init__()
        self.semaphore = semaphore
        if encode_save:
            if name:
                self.base_path = f"encode_files_{name}/{base_path}/"
            else:
                self.base_path = "encode_files/{}/".format(base_path)
        else:
            self.base_path = base_path + '/'
            
        self.temp_path = self.base_path + "temp/"
        self.bitstream_path = self.base_path + "bit/"
        self.recon_path = self.base_path + "recon/"
        self.img_path = img_path
        if not 0 <= qp <= 51:
            raise ValueError(f"Invalid quality value: {qp} (0,51)")
        self.qp = qp
        self.is_comp = is_comp
        
        base_dir = "static_file"
        self.bpg_encoder_path = "{}/bpgenc".format(base_dir)
        self.bpg_decoder_path = "{}/bpgdec".format(base_dir)
        
        self.bitdepth = "8"
        self.subsampling_mode = "444"
        # self.color_mode = "rgb"
        self.color_mode = "ycbcr"
        # self.encoder = "x265"
        self.encoder = "jctvc"
        
        self.lock = threading.RLock()
        
    def check_file(self, clear=False):
        self.checkdir(self.base_path, clear)
        self.checkdir(self.bitstream_path, clear)
        self.checkdir(self.recon_path, clear)
    
    def run(self):
        # the function runs when the entry point start() method called
        # thread_name = threading.current_thread().name
        # print("\ncurrent thread name: {}".format(thread_name))
        self.semaphore.acquire()
        self.lock.acquire()
        img_name = os.path.basename(self.img_path).split(".")[0]
        bit_path = os.path.join(self.bitstream_path, f"{img_name}.bin")
        rec_path = os.path.join(self.recon_path, f"{img_name}.png")
        
        encode_cmd = [
            self.bpg_encoder_path,
            "-o",
            bit_path,
            "-q",
            str(self.qp),
            "-f",
            self.subsampling_mode,
            "-e",
            self.encoder,
            "-c",
            self.color_mode,
            "-b",
            self.bitdepth,
            self.img_path,
        ]

        subp.call(encode_cmd, shell=False)
        
        decode_cmd = [
            self.bpg_decoder_path, "-o", rec_path, bit_path
        ]
        
        subp.call(decode_cmd, shell=False)
        
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


def run_bpg():
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
        t = BpgEncodeWorker(img_path, f"bpg_{args.quality}", args.quality, semaphore, is_comp=False, encode_save=True)
        threads_list.append(t)
        
    t.check_file(clear=True)
    
    pbar = tqdm.tqdm(total=len(threads_list))
    pbar.set_description("bpg encoding")
    
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
    run_bpg()
    