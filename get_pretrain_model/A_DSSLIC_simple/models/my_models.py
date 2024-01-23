import numpy as np
import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from PIL import Image
import pytorch_msssim
import util.util as util
import os
import time
import torchvision.transforms as transforms
from .vvc_encoding_model import run_train_encoder
import datetime
from collections import OrderedDict
import logging
from .unet_parts import DoubleConv

from data.base_dataset import get_transform, get_params
from .mmedit_super_resolution import RDN


class BaseMyModel(BaseModel):
    def __init__(self):
        super().__init__()
    
    def name(self):
        return "BaseMymodel"
    
    def initialize(self, opt, logger, comp_dir):
        raise NotImplementedError("sub class should define the method")
    
    def forward(self, input):
        raise NotImplementedError("sub class should define the method")
        
    def save_network(self, network, label, network_label, gpu=True):
        save_filename = '%s_%s.pth' % (label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if gpu and torch.cuda.is_available():
            network.cuda()
    
    def load_network(self, network, network_label=None, label=None, save_dir='', logger=None, save_path=None):        
        if not save_path:
            assert network_label is not None
            assert label is not None
            save_filename = '%s_%s.pth' % (label, network_label)
            if not save_dir:
                save_dir = self.save_dir
            print(save_dir)
            
            if logger:
                logger.info("load model from {}/{}".format(save_dir, save_filename))
            
            save_path = os.path.join(save_dir, save_filename)  
        else:
            if logger:
                logger.info("load model from {}".format(save_path))
              
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            raise FileExistsError("con not find the weights of the network")
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:   
                try:
                    state_dict_saved = torch.load(save_path)
                    state_dict_net = OrderedDict()
                    
                    for k, v in state_dict_saved.items():
                        state_dict_net[k.replace("module.", "")] = v
                    
                    network.load_state_dict(state_dict_net)
                except:
                    pretrained_dict = torch.load(save_path)                
                    model_dict = network.state_dict()
                    try:
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                        network.load_state_dict(pretrained_dict)
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                    except:
                        print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                        try:
                            from sets import Set
                            not_initialized = Set()
                        except:
                            not_initialized = set()
                        for k, v in pretrained_dict.items():                      
                            if v.size() == model_dict[k].size():
                                model_dict[k] = v

                        for k, v in model_dict.items():
                            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                                not_initialized.add(k.split('.')[0])                            
                        print(sorted(not_initialized))
                        network.load_state_dict(model_dict) 
    
    @classmethod
    def load_pretrain_compress_network(cls, network, save_path):
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            state_dict_saved = torch.load(save_path)
            state_dict_net = OrderedDict()
            
            for k, v in state_dict_saved.items():
                state_dict_net[k.replace("module.", "")] = v
            
            network.load_state_dict(state_dict_net)


class AutoEncoderWinAttn(BaseMyModel):
    def __init__(self, norm='instance', data_range=127, _type="noise"):
        """_summary_
        Args:
            norm (str, optional): instance or batch. Defaults to 'instance'.
        """
        super().__init__()
        self.norm = norm
        self.range = data_range
        self.type = _type
    
    def name(self):
        return "AutoencoderWinAttn"
    
    def initialize(self, opt, logger, comp_dir, no_gpu=False, load_dir=None, clamp=False):
        if logger:
            self.logger = logger
        self.istrain = opt.train
        self.save_dir = opt.save_dir
        self.train_comp = opt.finetune_comp
        if opt.resize_or_crop != 'none':  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        
        if self.norm == "instance":
            if self.type == "noise":
                self.compress_net = networks.CompressAutoEncoderWinAttnGelu(
                    input_nc=3,
                    output_nc=3,
                    ngf=64,
                    n_downsampling=4,
                    clamp=clamp,
                    d_range=self.range,
                    norm_layer=networks.get_norm_layer('instance')
                )
            assert self.type == "ste"
            self.compress_net = networks.CompressAutoEncoderWinAttnGeluSTE(
                3,
                3,
                d_range=self.range,
                norm_layer=networks.get_norm_layer('instance'),
            )
        else:
            raise NotImplementedError("No more norm layer")
        
        self.upsample = torch.nn.Upsample(scale_factor=16)
        if not no_gpu:
            self.compress_net.cuda()
        
        if not opt.train:
            if not load_dir:
                self.load_network(self.compress_net, "compress_net", "best_epoch_loss", logger=logger)
            else:
                self.load_network(self.compress_net, save_path=load_dir)
            
        print('---------- Networks initialized -------------')
        
        if logger:
            logger.info("\n\n\n--------------------------new running------------------------------------------"\
                "\n\n\nCompression network is : {}".format(self.compress_net.__repr__()))
            logger.info('---------- Networks initialized -------------')
        
        if opt.continue_train and opt.train:
            self.load_network(self.compress_net, "compress_net", "latest1", logger=logger)       
        
    def forward(self, input):
        if self.istrain:
            fake = self.compress_net(input)
            return fake
        
        else:
            with torch.no_grad():
                fake, comp_color, comp = self.compress_net.inference(input)
                up = self.upsample(comp)
                return input, up.repeat(1, 3, 1, 1), comp_color, fake, comp

    def compress(self, real):
        return self.compress_net.compress(real)
    
    def decompress(self, comp):
        return self.compress_net.decompress(comp)

