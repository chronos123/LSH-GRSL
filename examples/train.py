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

import argparse
import random
import shutil
import sys
import datetime
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import re

from torch.utils.data import DataLoader, sampler
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.zoo import image_models
import compressai.models.google as my_models
import util
import tqdm
from collections import OrderedDict


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, writer=None,
    norm=True
):
    model.train()
    device = next(model.parameters()).device
    
    epoch_mse_loss = 0
    epoch_bpp_loss = 0
    epoch_ms_ssim_loss = 0
    epoch_loss = 0
    psnr_1 = 0
    num = 0
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        if criterion.metric == "mse":
            epoch_mse_loss += out_criterion["mse_loss"].item()
        else:
            epoch_ms_ssim_loss +=  out_criterion["ms_ssim_loss"].item()
            
        epoch_bpp_loss += out_criterion["bpp_loss"].item()
        epoch_loss += out_criterion["loss"].item()
        
        if "fake" in out_net.keys():
            fake = out_net["fake"]
            psnr_1 += util.psnr_batch_all(fake, d, norm=norm)
        else:
            psnr_1 = 0
            
        num += 1
        bs = d.shape[0]
        # epoch_aux_loss = aux_loss / bs
        if i % 10 == 0:
            if criterion.metric == "mse":
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |\n'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.5f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f} |"
                    f"\tpsnr fake mean: {psnr_1 / (num * bs): .2f}"
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |\n'
                    f'\tms_ssim loss: {out_criterion["ms_ssim_loss"].item():.5f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f} |"
                    f"\tpsnr fake mean: {psnr_1 / (num * bs): .2f}"
                )
    
    if writer:
        divide_ratio = len(train_dataloader)
        if criterion.metric == "mse":
            writer.add_scalar("mse_loss", epoch_mse_loss / divide_ratio, epoch)
        else:
            writer.add_scalar("ms_ssim-loss", epoch_ms_ssim_loss/divide_ratio, epoch)
        writer.add_scalar("bpp_loss", epoch_bpp_loss / divide_ratio, epoch)
        if "fake" in out_net.keys():
            writer.add_scalar("psnr_fake", psnr_1 / (num * bs), epoch)
        writer.add_scalar("loss", epoch_loss / divide_ratio, epoch)
        # writer.add_scalar("aux_loss", aux_loss.item(), epoch)


def test_epoch(epoch, test_dataloader, model, criterion, writer=None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    image_quality_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if criterion.metric == "mse":
                image_quality_loss.update(out_criterion["mse_loss"])
            else:
                image_quality_loss.update(out_criterion["ms_ssim_loss"])
                
    if criterion.metric == "mse":
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {image_quality_loss.avg:.5f} |"
            f"\tBpp loss: {bpp_loss.avg:.4f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )
    else:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tms_ssim loss: {image_quality_loss.avg:.5f} |"
            f"\tBpp loss: {bpp_loss.avg:.4f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    if writer:
        if criterion.metric == "mse":
            writer.add_scalar("mse_loss_val", image_quality_loss.avg, epoch)
        else:
            writer.add_scalar("ms_ssim_loss_val", image_quality_loss.avg, epoch)
            
        writer.add_scalar("bpp_loss_val", bpp_loss.avg, epoch)
        writer.add_scalar("loss_val", loss.avg, epoch)
    
    return loss.avg, image_quality_loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", best_file="checkpoint_best_loss.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", 
        "--dataset",
        type=str, 
        default="/media/disk2/sxc/DSSLIC/datasets/NASA-final",
        help="Training dataset"
    )
    
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    
    parser.add_argument(
        "--new_lrsch",
        action="store_true"
    )
    
    parser.add_argument(
        "--new_lr",
        action="store_true",
        help="do not continue training with the same lr"
    )
    
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=2,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--metric", type=str, default="mse", help="mse or ms_ssim")
    args = parser.parse_args(argv)
    return args


def main(argv):
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    args = parse_args(argv)
    fine_001 = False
    quality = 1
    # qulity=4 (cheng), qulity=2 (ours)
    
    if args.model not in [
        "bmshj2018-factorized",
        "bmshj2018-factorized_relu",
        "bmshj2018-hyperprior",
        "mbt2018",
        "mbt2018-mean",
        "cheng2020-anchor",
        "cheng2020-attn",
        ]:
        model_tag = f"{args.model}_res_lambda{args.lmbda}_lr{args.learning_rate}"
        finetune = True
        norm = False
        
        if finetune:
            model_tag += "_finetune"
            # model_tag += "_flip"
        if norm:
            model_tag += "_normalize"
        
        if args.checkpoint:
            if len(re.findall("pre_trained_model", args.checkpoint)) != 0:
                model_tag += "_start_pretrained"
        # model_tag += "_ssim-pretrained"
        net = image_models[args.model](quality=quality, finetune=finetune, norm=norm)

        # model_tag += "_flip"

    else:
        finetune = False
        norm = False            
        model_tag = f"{args.model}_lambda{args.lmbda}_lr{args.learning_rate}"
        if hasattr(args, "checkpoint"):
            if args.checkpoint:
                if len(re.findall("pre_trained_model", args.checkpoint)) != 0:
                    model_tag += "_start_pretrained"
        net = image_models[args.model](quality=quality)
        
    model_tag += f"_q{quality}"
    # net = image_models[args.model](quality=1)        
    # quality 1: low bitrate 
    # quality 2: high bitrate
    # net = my_models.HyperprioiorResEncoder(N=128, M=192, finetune=finetune, norm=norm)
    if fine_001:
        if hasattr(net, "set_001fine"):
            net.set_001fine()

    dataset_name = os.path.basename(args.dataset)
    base_path = f"./checkpoints_{dataset_name}_psnr"
    if args.metric == "ms_ssim":
        base_path = f"./checkpoints_{dataset_name}_ms_ssim"
    
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    
    save_dir = f"{base_path}/{args.epochs}_{model_tag}_size-{args.patch_size[0]}x{args.patch_size[1]}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        os.mkdir(f"{save_dir}/tensorboard")
        
    ckp_path = f"{save_dir}/checkpoint.pth.tar"
    best_ckp_path = f"{save_dir}/checkpoint_best_loss.pth.tar"
    if args.metric == "mse":
        best_quality_path = f"{save_dir}/checkpoint_best_mse.pth.tar"
    else:
        best_quality_path = f"{save_dir}/checkpoint_best_ms_ssim.pth.tar"
    
    tensorboard_path = f"{save_dir}/tensorboard/{time}"
    summary_writer = SummaryWriter(tensorboard_path)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if norm:
        train_transforms = transforms.Compose(
            [transforms.RandomCrop(args.patch_size), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        test_transforms = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                # transforms.RandomRotation((0, 135)),
                transforms.RandomCrop(args.patch_size), 
                transforms.ToTensor()
             ]
        )

        test_transforms = transforms.Compose(
            [
                # transforms.CenterCrop((1024, 1024)),
                transforms.ToTensor(),
            ]
        )
        
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

# split = val / test
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda")
    )

    # net = image_models[args.model](quality=1)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
    if args.metric == "ms_ssim":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5)
    else:
        if args.new_lrsch:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs*0.9), gamma=0.1)
        else:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=5)
    # add factor and patience
    
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric=args.metric)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "epoch" in  checkpoint.keys():
            last_epoch = checkpoint["epoch"] + 1
            try:
                net.load_state_dict(checkpoint["state_dict"])
            except:
                net.from_state_dict(checkpoint["state_dict"])
            if args.new_lr:
                print(args.learning_rate)
                aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            else:
                optimizer.load_state_dict(checkpoint["optimizer"])
                aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        else:
            try:
                net.load_state_dict(checkpoint)
                print("loaded")
            except:
                state_dick = OrderedDict()
                for k, v in checkpoint.items():
                    if "entropy_bottleneck._biases" in k:
                        k = k.replace("biases.", "bias")
                    if "entropy_bottleneck._matrices" in k:
                        k = k.replace("matrices.", "matrix")    
                    if "entropy_bottleneck._factors" in k:
                        k = k.replace("factors.", "factor")
                    state_dick[k] = v 
                net.from_state_dict(state_dick)

    best_loss = float("inf")
    best_mse_loss = float("inf")
    pbar = tqdm.tqdm(total=args.epochs - last_epoch, desc="train")
    
    for epoch in range(last_epoch, args.epochs):
        # lr_scheduler.step(epoch)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            writer=summary_writer,
            norm=norm
        )
        # lr_scheduler.step()
        if epoch % 3 == 0:
            loss, mse_loss = test_epoch(epoch, test_dataloader, net, criterion, writer=summary_writer)
            lr_scheduler.step(loss)
        
            is_best = loss < best_loss
            is_image_quality_best = mse_loss < best_mse_loss
            best_loss = min(loss, best_loss)
            best_mse_loss = min(mse_loss, best_mse_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    filename=ckp_path,
                    best_file=best_ckp_path
                )
                
            if is_image_quality_best:
                shutil.copy(ckp_path, best_quality_path)  
                
        pbar.update(1)

    pbar.close()
    print(f"finetune {finetune}, norm {norm}")


if __name__ == "__main__":
    main(sys.argv[1:])
