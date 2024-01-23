import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.aligned_dataset import AlignedDataset, make_dataset

import os.path
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform_comp, normalize
from data.image_folder import make_dataset
from PIL import Image
import util.util as util
import torch
import os


class CompImageSet(AlignedDataset):    
    def name(self):
        return "CompImageSet"
    
    def initialize(self, opt, img_path):
        """_summary_

        Args:
            img_path (str): the directory containing all the comp_image files
        """
        self.opt = opt
        
        ### label maps

        ### real images
        #if opt.isTrain:
        self.dir_image = img_path
        self.image_paths = sorted(make_dataset(self.dir_image, opt))
        
        # make the #images divisible by batch size
        numImg = int(len(self.image_paths)/opt.batchSize)*opt.batchSize        
        self.image_paths = self.image_paths[0:numImg]     
     
        self.dataset_size = len(self.image_paths) 

    def __getitem__(self, index):   
        ## TODO: rewrite the transform to solve the shape distortion problem       
        image_tensor = ds_tensor = inst_tensor = feat_tensor = 0
        ### real images
        #if self.opt.isTrain:
        image_path = self.image_paths[index]           
        image = Image.open(image_path).convert('RGB')
        # print(image.width)
        params = get_params(self.opt, image.size)
        transform_image = get_transform_comp(self.opt, params)      
        image_tensor = transform_image(image)        
        
        # transform reshape the comp_image
        
        # print(image_tensor.shape)
        label_tensor=0
          
        input_dict = {'label': label_tensor, 'inst': inst_tensor, 'image': image_tensor, 'ds': ds_tensor,
                        'feat': feat_tensor, 'path': image_path}

        return input_dict


def CreateDataset(opt, img_path):
    dataset = None
    dataset = CompImageSet()   

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt, img_path)
    return dataset


class CompImageDataLoader(BaseDataLoader):
    def name(self):
        return "MarsDatasetDataLoader"
    
    def initialize(self, opt, img_path):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, img_path)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
