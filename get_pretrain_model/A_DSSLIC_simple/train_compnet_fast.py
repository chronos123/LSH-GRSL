### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
## speed up 4 times

from options.train_options_mars import ParameterFinenet
from data.data_loader import create_dataloader_mars, create_dataloader_mars_nonorm
import models.my_models as model_set
import util.util as util
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6, 7"
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import logging
import tqdm
import pytorch_msssim
from copy import deepcopy
from torch.optim import lr_scheduler
import datetime
from torch import nn
from argparse import ArgumentParser

from utility import psnr_array
from calc_ssim import _calc_ssim_array


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_loss(pred, ground_truth, main_loss_fn, epoch):
    loss = main_loss_fn(pred, ground_truth) 
    # if epoch >= 20:
    #     loss += 5 * models.pytorch_msssim.ms_ssim(
    #                 (real_image + 1) / 2,
    #                 (recon + 1) / 2,
    #                 data_range=1
    #             )
    return torch.clamp(loss, 1e-7, 1e5)


def calc_metrics(tensor1, tensor2):
    img1 = util.tensor2im(tensor1)
    img2 = util.tensor2im(tensor2)
    return psnr_array(img1, img2), _calc_ssim_array(img1, img2)
    

def ms_ssim_loss(x, y):
    x1 = (x + 1) / 2
    y1 = (y + 1) / 2
    return 1 - pytorch_msssim.ms_ssim(x1, y1, data_range=1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, type=str)

    args = parser.parse_args()
    total_epoch = 500
    batch_size = 2
    
    continue_train = False
    scheduler = True
    clamp = True
    
    model_name = "AutoEncoderWinAttn"
    model = model_set.AutoEncoderWinAttn(data_range=1, _type="ste")
    no_test = False
    
    # loss_fn = torch.nn.L1Loss()
    loss_fn = nn.MSELoss()
    # loss_fn = ms_ssim_loss
    comp_pretrian_dir = None
    
    if not scheduler:
        base_lr = 0.0002
        descend_ep = 100
        epochs = list(range(50, total_epoch + 1, descend_ep))
        lrs = [base_lr * 0.7**i for i in range(0, int(total_epoch / descend_ep) - 1)]
        lr_dict = dict(zip(epochs, lrs))
    else:
        base_lr = 0.0002
    
    beta1 = 0.5
    # default
    
    opt = ParameterFinenet(
        ckp_dir="comp_network",
        name=model_name,
        batchsize=batch_size,
        data_root=args.dataset,
        continue_train=continue_train,
        random_flip=True,
        choice = "scalecrop",
        # size=1200,
        scalecropsize=(512, 512),
        finetune_comp=False
        )
    # [resize_and_crop|crop|scale_width|scale_width_and_crop|scalecrop]
    
    # opt.max_dataset_size = 4
    opt.nThreads = 8
    data_loader = create_dataloader_mars(opt)

    logging.basicConfig(
        filename=opt.log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode="a+"
        )
    
    logger = logging.getLogger("model_summarize")
    
    model.initialize(opt, logger, comp_dir=comp_pretrian_dir, clamp=clamp)

    logger.info("model {} is trained, total epoch {}, batch size {}".format(model.name(), total_epoch, batch_size))
    
    with open("{}/info.txt".format(opt.log_dir), "w") as file:
        file.write('------------ lr -------------\n')
        file.write("base learn rate" + str(base_lr) + "\n")
        file.write(f"scheduler is {scheduler}\n")
        if hasattr(model, "range"):
            file.write(f"data range is {model.range}\n")
        if not scheduler:
            for k, v in sorted(lr_dict.items()):
                file.write('%s: %s' % (str(k), str(v)))
                file.write("\n")
        file.write("\n")
        file.write('------------ Options -------------\n')
        for k, v in sorted(opt.__dict__.items()):
            file.write('%s: %s' % (str(k), str(v)))
            file.write("\n")
        file.write('-------------- End ----------------\n')
    
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    summary_writer = SummaryWriter("{}/{}/tensorboard/{}".format(opt.ckp_dir, opt.name, time))

    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    
    opt1 = deepcopy(opt)
    opt1.phase = "val"
    opt1.train = False
    opt1.isTrain = False
    opt1.resize_or_crop = "scale_width"
    opt1.random_flip = False
    opt1.serial_batches = True
    opt1.how_many = 386
    opt1.ntest = 386
    opt1.batchSize = 1
    opt1.loadSize = 1600
    data_loader_valid = create_dataloader_mars(opt1)
    dataset_valid = data_loader_valid.load_data()
    valid_size = len(data_loader_valid)
    print('#validation images = %d' % valid_size)
    
    with open("{}/valid_info.txt".format(opt.log_dir), "w") as file:
        file.write('------------ Options -------------\n')
        for k, v in sorted(opt1.__dict__.items()):
            file.write('%s: %s' % (str(k), str(v)))
            file.write("\n")
        file.write('-------------- End ----------------\n')
    
    device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))))
    net = torch.nn.DataParallel(model, device_ids=device_ids)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr, betas=(beta1,0.999))
    
    if scheduler:
        lr_sch = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)
    
    best_epoch_loss = float('inf')
    best_epoch_ssim = 0
    best_epoch_psnr = 0
    best_valid_psnr = 0
    best_valid_ssim = 0
    
    if continue_train:
        continue_epoch = int(input("coninue epoch: "))
        best_epoch_loss = float(input("best loss: "))
        best_epoch_psnr = float(input("best psnr: "))
        best_epoch_ssim = float(input("best ssim: "))
    
    for epoch in tqdm.tqdm(list(range(0, total_epoch+1))):
        
        epoch_loss = 0
        psnrs = []
        ssims = []
        # ms_ssims = []
        
        if not scheduler:
            if epoch in lr_dict.keys():
                update_lr(optimizer, lr_dict[epoch])
                if hasattr(net.module, "finenet"):
                    model.save_network(net.module.finenet, "{}".format(epoch), "finenet")
                model.save_network(net.module.compress_net, "{}".format(epoch), "compress_net")
        else:
            print(f"learning rate is {lr_sch.get_last_lr()}")
        
        if continue_train:
            if epoch < continue_epoch:
                print("epoch {} skip".format(epoch))
                continue
            
        for i, data in enumerate(dataset):
            if model_name == "test_code":
                if i > 2:
                    break
                print(i)
                
            real_image = Variable(data['image'].cuda())
            recon = net(real_image)

            loss = get_loss(real_image, recon, main_loss_fn=loss_fn, epoch=epoch)
            epoch_loss += loss.item()
            
            if (i + 1)*batch_size % 1000 == 0:
                # print(i)
                if hasattr(net.module, "finenet"):
                    model.save_network(net.module.finenet, "latest1", "finenet")
                model.save_network(net.module.compress_net, "latest1", "compress_net")
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if scheduler:
            lr_sch.step()
        
        epoch_loss = epoch_loss / (dataset_size / batch_size)
        
        # psnr_epoch = np.mean(np.array(psnrs))
        # ssim_epoch = np.mean(np.array(ssims))
        # ms_ssim_mean = np.mean(np.array(ms_ssims))

        summary_writer.add_scalar("epoch_loss", epoch_loss, epoch)
        # summary_writer.add_scalar("psnr", psnr_epoch, epoch)
        # summary_writer.add_scalar("ssim", ssim_epoch, epoch)
        # summary_writer.add_scalar("ms_ssim", ms_ssim_mean, epoch)
        
        # summary_writer.add_images("real", real_image, 0)
        # summary_writer.add_images("recon", recon, 0)

        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            if hasattr(net.module, "finenet"):
                model.save_network(net.module.finenet, "best_epoch_loss", "finenet")
            model.save_network(net.module.compress_net, "best_epoch_loss", "compress_net")
        
        # if psnr_epoch > best_epoch_psnr:
        #     best_epoch_psnr = psnr_epoch
        #     if hasattr(net.module, "finenet"):
        #         model.save_network(net.module.finenet, "best_epoch_psnr", "finenet")
        #     model.save_network(net.module.compress_net, "best_epoch_psnr", "compress_net")
            
        # if ssim_epoch > best_epoch_ssim:
        #     best_epoch_ssim = ssim_epoch
        #     if hasattr(net.module, "finenet"):
        #         model.save_network(net.module.finenet, "best_epoch_ssim", "finenet")
        #     model.save_network(net.module.compress_net, "best_epoch_ssim", "compress_net")
        
        logger.info(f"epoch {epoch} loss: {epoch_loss:.7f}")
        
        if no_test:
            continue
        
        if (epoch + 1) % 30 == 0:
            psnrs_valid = []
            ssims_valid = []
            for i, data in enumerate(dataset_valid):
                if model_name == "test_code":
                    if i > 2:
                        break
                    print(i)
                    
                with torch.no_grad():
                    real_image = Variable(data['image'].cuda())
                
                    recon = net(real_image)

                    for j in range(1):
                        psnr, ssim = calc_metrics(real_image.data[j], recon.data[j])
                        psnrs_valid.append(psnr)
                        ssims_valid.append(ssim)
            
            psnr_valid = np.mean(np.array(psnrs_valid))
            ssim_valid = np.mean(np.array(ssims_valid))
            
            if psnr_valid > best_valid_psnr:
                best_valid_psnr = psnr_valid
                if hasattr(net.module, "finenet"):
                    model.save_network(net.module.finenet, "best_valid_psnr", "finenet")
                model.save_network(net.module.compress_net, "best_valid_psnr", "compress_net")
                
            if ssim_valid > best_valid_ssim:
                best_valid_ssim = ssim_valid
                if hasattr(net.module, "finenet"):
                    model.save_network(net.module.finenet, "best_valid_ssim", "finenet")
                model.save_network(net.module.compress_net, "best_valid_ssim", "compress_net")
            
            summary_writer.add_scalar("psnr_valid", psnr_valid, epoch)
            summary_writer.add_scalar("ssim_valid", ssim_valid, epoch)
            
            # logger.info("epoch {0} loss: {1:.7f}, \nbest_psnr: {2:.2f}, best_ssim: {3:.4f}, " \
            #     "best loss {4:.7f}\nbest_valid_psnr: {5:.2f}, best_valid_ssim: {6:.4f}".format(
            #     epoch, epoch_loss, best_epoch_psnr, best_epoch_ssim, best_epoch_loss,
            #     best_valid_psnr, best_valid_ssim))
            
            
            
        
                
