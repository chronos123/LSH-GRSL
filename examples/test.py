# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Evaluate an end-to-end compression model on an image dataset.
"""

import argparse
import json
import math
import sys
import time
import tqdm
import os
import subprocess
import cv2

from io import BytesIO
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
from multiprocessing import Pool, Lock

import compressai
from compressai.zoo import image_models as pretrained_models
from compressai.zoo.image import model_architectures as architectures
import util
from copy import deepcopy
import datetime

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

_temp_file_name = ""

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    image_files = []
    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).glob(f"*{ext}"))
    return image_files


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, max_val: int = 255
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def read_image(filepath: str) -> torch.Tensor:
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    global normalize
    if normalize:
        transforms_list = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transforms_list = transforms.Compose([
            transforms.ToTensor(),
        ])
    return transforms_list(img)


@torch.no_grad()
def inference(model, x, filepath=None, time_1=None):
    global multi
    if multi:
        lock = Lock()
        lock.acquire()
    else:
        lock = Lock()
        lock.acquire()
    
    if time_1:
        file_tag = model.__name__ + time_1
    else:
        file_tag = model.__name__
    if multi:
        if filepath:
            multi_tag = filepath.stem
        else:
            multi_tag = "1"
    else:
        multi_tag = "0"
    
    if not os.path.exists(f"temp{file_tag}"):
        os.makedirs(f"temp{file_tag}")
    fp = open(f"temp{file_tag}/temp{multi_tag}.txt", "w")
    global _temp_file_name
    _temp_file_name = f"temp{file_tag}"
    
    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], out_enc["comp"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    
    global i, pbar, normalize, save
    
    if save:
        os.makedirs(f"./test_one/{model.__name__}", exist_ok=True)
        if not multi:
            # for debug
            real_image = util.tensor2im(x.data[0], normalize=normalize)
            Image.fromarray(real_image).save(
                f"./test_one/{model.__name__}/{i}_real.png"
                )
            out_image = util.tensor2im(out_dec["x_hat"].data[0], normalize=normalize)
            Image.fromarray(out_image).save(
                f"./test_one/{model.__name__}/{i}_rec.png"
                )
            fake_image = util.tensor2im(out_dec["fake"].data[0], normalize=normalize)
            Image.fromarray(fake_image).save(
                f"./test_one/{model.__name__}/{i}_fake.png"
                )
            highfrq_image = util.tensor2im(out_dec["x_hat"].data[0] - out_dec["fake"].data[0], normalize=normalize)
            Image.fromarray(highfrq_image).save(
                f"./test_one/{model.__name__}/{i}_highfrq.png"
                )
            highfrq_real_image = util.tensor2im(x.data[0] - out_dec["fake"].data[0], normalize=normalize)
            Image.fromarray(highfrq_real_image).save(
                f"./test_one/{model.__name__}/{i}_highfrq_real.png"
                )
        else:
            os.makedirs(f"./test_one/recon_image", exist_ok=True)
            real_image = util.tensor2im(x.data[0], normalize=normalize)
            Image.fromarray(real_image).save(
                f"./test_one/{model.__name__}/{multi_tag}_{i}_real.png"
                )
            out_image = util.tensor2im(out_dec["x_hat"].data[0], normalize=normalize)
            Image.fromarray(out_image).save(
                f"./test_one/{model.__name__}/{multi_tag}_{i}_rec.png"
                )
            fake_image = util.tensor2im(out_dec["fake"].data[0], normalize=normalize)
            Image.fromarray(fake_image).save(
                f"./test_one/{model.__name__}/{multi_tag}_{i}_fake.png"
                )
            highfrq_image = util.tensor2im(out_dec["x_hat"].data[0] - out_dec["fake"].data[0], normalize=normalize)
            Image.fromarray(highfrq_image).save(
                f"./test_one/{model.__name__}/{multi_tag}_{i}_highfrq.png"
                )
            highfrq_real_image = util.tensor2im(x.data[0] - out_dec["fake"].data[0], normalize=normalize)
            Image.fromarray(highfrq_real_image).save(
                f"./test_one/{model.__name__}/{multi_tag}_{i}_highfrq_real.png"
                )
    
    out_dec["fake"] = F.pad(
        out_dec["fake"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    
    metrics_f = compute_metrics(x, out_dec["fake"], 255)
    
    if not multi:
        i += 1    
        pbar.update(1)
    
    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    
    # output = BytesIO()
    # image_output_c = Image.fromarray(util.tensor2im(out_enc["comp"].data[0]))
    # image_output_c.save(output, 'PNG') #a format needs to be provided
    # image_filesize  = output.tell()
    # output.close()
    # comp_bpp = image_filesize * 8.0 / num_pixels
    
    data_range = model.pre_encoder.data_range
    
    one_comp_arr = out_enc["comp"].round().detach().cpu().numpy()
    one_comp_arr = one_comp_arr[0, :, :, :]
    one_comp_arr = np.transpose(one_comp_arr, (1, 2, 0))
    one_comp_arr = one_comp_arr + data_range
    if data_range > 127:
        cv2.imwrite(f"temp{file_tag}/comp_full_temp{multi_tag}.png", one_comp_arr.astype(np.uint16))
    else:
        cv2.imwrite(f"temp{file_tag}/comp_full_temp{multi_tag}.png", one_comp_arr.astype(np.uint8))
    if save:
        if not multi:
            cv2.imwrite(
                f"./test_one/{model.__name__}/{i-1}_feature.png",
                one_comp_arr.astype(np.uint8) * 51
                    )
        else:
            cv2.imwrite(
                f"./test_one/{model.__name__}/{multi_tag}_{i-1}_feature.png",
                one_comp_arr.astype(np.uint8) * 51
                    )
    # im = cv2.imread("comp_full_temp.png", cv2.IMREAD_ANYDEPTH)
    # im = im.astype(np.float32) - 32767
    # im = torch.from_numpy(im).cuda()
    # im.unsqueeze_(0)
    # im.unsqueeze_(0)
    # up_image = model.decompress(out_enc["strings"], out_enc["shape"], im)
    
    flif_path = "./examples/flif"
    subprocess.call(
        [flif_path, '-e', f"temp{file_tag}/comp_full_temp{multi_tag}.png", f'temp{file_tag}/comp{multi_tag}.flif', "--overwrite"],
        stderr=fp,
        stdout=fp
        )
    subprocess.call(
        [flif_path, '-d', f'temp{file_tag}/comp{multi_tag}.flif', f"temp{file_tag}/comp_recon{multi_tag}.png", "--overwrite"],
        stdout=fp,
        stderr=fp
        )
    # comp_bpp = util.calc_bpp_vvc_exp(f"temp{file_tag}/comp{multi_tag}.flif", choice="mars")
    comp_bpp = os.stat(f"temp{file_tag}/comp{multi_tag}.flif").st_size * 8.0 / num_pixels
    fp.close()
    
    subprocess.run(["rm", f"temp{file_tag}/comp_full_temp{multi_tag}.png", f"temp{file_tag}/comp{multi_tag}.flif",
                    f"temp{file_tag}/comp_recon{multi_tag}.png", f"temp{file_tag}/temp{multi_tag}.txt"])
    
    bpp += comp_bpp
    
    if save:
        with open(f"./test_one/{model.__name__}/{i-1}_bpp.txt", "w") as file:
            file.write(f"bpp is: {bpp: .4f}")
    
    if multi:
        lock.release()
    else:
        lock.release()
    
    return {
        "psnr_fake": metrics_f["psnr"],
        "psnr": metrics["psnr"],
        "ms-ssim": metrics["ms-ssim"],
        "bpp": bpp,
        "comp_bpp": comp_bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_net["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr": metrics["psnr"],
        "ms-ssim": metrics["ms-ssim"],
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()


def load_checkpoint(arch: str, no_update: bool, checkpoint_path: str) -> nn.Module:
    # update model if need be
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    # compatibility with 'not updated yet' trained nets
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]

    model_cls = architectures[arch]
    net = model_cls.from_state_dict(state_dict)
    if not no_update:
        net.update(force=True)
    return net.eval()


def eval_model(
    model: nn.Module,
    outputdir: Path,
    filepaths,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for filepath in filepaths:

        x = read_image(filepath).to(device)
        if not entropy_estimation:
            if args["half"]:
                model = model.half()
                x = x.half()
            rv = inference(model, x)
        else:
            rv = inference_entropy_estimation(model, x)
        for k, v in rv.items():
            metrics[k] += v
        if args["per_image"]:
            if Path(outputdir).is_dir():
                image_metrics_path = (
                    Path(outputdir) / f"{filepath.stem}-{trained_net}.json"
                )
                with image_metrics_path.open("wb") as f:
                    output = {
                        "source": filepath.stem,
                        "name": args["architecture"],
                        "description": f"Inference ({description})",
                        "results": rv,
                    }
                    f.write(json.dumps(output, indent=2).encode())
            else:
                raise FileNotFoundError("Please specify output directory")

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def eval_process(
    filepath,
    device,
    model,
    outputdir,
    entropy_estimation,
    trained_net,
    description,
    time_1,
    args
):
    metrics = defaultdict(float)
    x = read_image(filepath).to(device)
    if not entropy_estimation:
        if args["half"]:
            model = model.half()
            x = x.half()
        rv = inference(model, x, filepath=filepath, time_1=time_1)
    else:
        rv = inference_entropy_estimation(model, x)
    for k, v in rv.items():
        metrics[k] += v
    if args["per_image"]:
        if Path(outputdir).is_dir():
            image_metrics_path = (
                Path(outputdir) / f"{filepath.stem}-{trained_net}.json"
            )
            with image_metrics_path.open("wb") as f:
                output = {
                    "source": filepath.stem,
                    "name": args["architecture"],
                    "description": f"Inference ({description})",
                    "results": rv,
                }
                f.write(json.dumps(output, indent=2).encode())
        else:
            raise FileNotFoundError("Please specify output directory")
        
    return metrics


def eval_model_multi(
    model: nn.Module,
    outputdir: Path,
    filepaths,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    n_threads = 20,
    **args: Any,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    metrics_all = defaultdict(float)
    time_1 = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    global pbar
    pbar.close()

    pool = Pool(n_threads)
    param_list = [
        (
            filepath,
            device,
            model,
            outputdir,
            entropy_estimation,
            trained_net,
            description,
            time_1,
            args
        ) for filepath in filepaths
    ]
    start = time.time()
    result_list = tqdm.tqdm(
        pool.starmap(eval_process, param_list),
        total=len(param_list),
        desc=f"process in {n_threads} threads"
        )
    consume = time.time() - start
    print(f"time used is {(consume/60): .2f} min")

    # print(len(result_list))
    # print(result_list)
    
    for r in result_list:
        for k, v in r.items():
            if k in metrics_all.keys():
                metrics_all[k] += v 
            else:
                metrics_all[k] = v
                
    # print(metrics_all)
    
    for k, v in metrics_all.items():
        metrics_all[k] = v / len(result_list)
    return metrics_all


def setup_args():
    # Common options.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    parent_parser.add_argument(
        "-k",
        "--kfold",
        type=int,
        default=-1,
    )
    parent_parser.add_argument(
        "-n",
        "--n_threads",
        type=int,
        default=30,
    )
    
    parent_parser.add_argument(
        "-d",
        "--output_directory",
        type=str,
        default="",
        help="path of output directory. Optional, required for output json file, results per image. Default will just print the output results.",
    )
    parent_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
        help="output json file name, (default: architecture-entropy_coder.json)",
    )
    parent_parser.add_argument(
        "--per-image",
        action="store_true",
        help="store results for each image of the dataset, separately",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="+",
        type=int,
        default=(1,),
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    checkpoint_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Disable the default update of the model entropy parameters before eval",
    )
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    global pbar
    if args.source not in ["checkpoint", "pretrained"]:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )

    if args.kfold != -1:
        raise NotImplementedError()
    else:
        filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    pbar.total = len(filepaths)
    
    compressai.set_entropy_coder(args.entropy_coder)

    # create output directory
    if args.output_directory:
        Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    if args.source == "pretrained":
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d}"
    else:
        runs = args.paths
        opts = (args.architecture, args.no_update)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s}"

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        model = load_func(*opts, run)
        if args.source == "pretrained":
            trained_net = f"{args.architecture}-{args.metric}-{run}-{description}"
        else:
            cpt_name = Path(run).name[: -len(".tar.pth")]  # removesuffix() python3.9
            trained_net = f"{cpt_name}-{description}"
        print(f"Using trained model {trained_net}", file=sys.stderr)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
        args_dict = vars(args)
        
        global normalize
        if hasattr(model, "norm"):
            model.norm = normalize
        
        global multi
        if args.cuda:
            print(f"cuda is used, multi = False")
            multi = False
        if args.n_threads == 1:
            multi = False
                
        if multi:
            metrics = eval_model_multi(
            model,
            args.output_directory,
            filepaths,
            trained_net=trained_net,
            description=description,
            **args_dict,
        )
        else:
            metrics = eval_model(
                model,
                args.output_directory,
                filepaths,
                trained_net=trained_net,
                description=description,
                **args_dict,
            )
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": f"{args.architecture}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }
    if args.output_directory:
        output_file = (
            args.output_file
            if args.output_file
            else f"{args.architecture}-{description}"
        )

        with (Path(f"{args.output_directory}/{output_file}").with_suffix(".json")).open(
            "wb"
        ) as f:
            f.write(json.dumps(output, indent=2).encode())
    pbar.close()
    print(json.dumps(output, indent=2))
    global _temp_file_name
    subprocess.call(
        [
            "rm",
            "-rf",
            _temp_file_name
        ]
        )


if __name__ == "__main__":
    try:
        i = 0
        pbar = tqdm.tqdm(total=386, desc="test")
        normalize = False
        multi = True
        save = False
        # multi = bool(int(input("multi is: 0-false, 1-true ")))
        # save = bool(int(input("save is : 0-false, 1-true ")))
        
        # n_threads = 30 
        
        print(f"normalize: {normalize}\nmulti: {multi}\nsave: {save}")
        if save:
            subprocess.check_call(
                "rm -f ./test_one/recon_image/*", 
                shell=True,
                )
        
        main(sys.argv[1:])
        os.system("rm temp*")
    except KeyboardInterrupt:
        os.system("rm temp*")
    
