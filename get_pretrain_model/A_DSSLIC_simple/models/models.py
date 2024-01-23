### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch

def create_model(opt):
    from .DSSLIC_model import DSSLICModel
    model = DSSLICModel()    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model


def create_decode_model(opt):
    from .DSSLIC_model import DSSLICModelWithVVC
    model = DSSLICModelWithVVC()    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model


def create_GDN_model(opt):
    from .DSSLIC_model import GDNModel
    model = GDNModel()    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model


def create_modified_vgg_model(opt):
    from .DSSLIC_model import DSSLICModefiedVGG
    model = DSSLICModefiedVGG()    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model


def create_modified_comp_model(opt):
    from .DSSLIC_model import DSSLICModifiedComp
    model = DSSLICModifiedComp()    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model


def create_loss_model(opt):
    from .DSSLIC_model import DSSLICModell1LossComp
    model = DSSLICModell1LossComp()    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model


def create_fine_model(opt):
    pass
