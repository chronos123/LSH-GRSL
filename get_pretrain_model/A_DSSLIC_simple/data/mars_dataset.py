import os.path
import os
from PIL import Image

from .aligned_dataset import AlignedDataset
from data.image_folder import make_dataset
from data.base_dataset import get_params, get_transform
from util import util

class MarsDataset(AlignedDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        
        ### label maps
        if not opt.no_seg:     
            print("label used is _label_ade_psp101")
            self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label_ade_psp101')              
            self.label_paths = sorted(make_dataset(self.dir_label,opt))

        ### real images
        #if opt.isTrain:
        self.dir_image = os.path.join(opt.dataroot, opt.phase)  
        self.image_paths = sorted(make_dataset(self.dir_image,opt))
        
        # make the #images divisible by batch size
        numImg = int(len(self.image_paths)/opt.batchSize)*opt.batchSize        
        self.image_paths = self.image_paths[0:numImg]    
        if not opt.no_seg:
            self.label_paths = self.label_paths[0:numImg]            

        ### instance maps
        if not opt.no_instance:
            raise ValueError("no instance map")

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat,opt))

        self.dataset_size = len(self.image_paths) 


class MarsDatasetNoNorm(AlignedDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        
        ### label maps
        if not opt.no_seg:     
            print("label used is _label_ade_psp101")
            self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label_ade_psp101')              
            self.label_paths = sorted(make_dataset(self.dir_label,opt))

        ### real images
        #if opt.isTrain:
        self.dir_image = os.path.join(opt.dataroot, opt.phase)  
        self.image_paths = sorted(make_dataset(self.dir_image,opt))
        
        # make the #images divisible by batch size
        numImg = int(len(self.image_paths)/opt.batchSize)*opt.batchSize        
        self.image_paths = self.image_paths[0:numImg]    
        if not opt.no_seg:
            self.label_paths = self.label_paths[0:numImg]            

        ### instance maps
        if not opt.no_instance:
            raise ValueError("no instance map")

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat,opt))

        self.dataset_size = len(self.image_paths) 
    
    def __getitem__(self, index):          
        image_tensor = ds_tensor = inst_tensor = feat_tensor = 0
        ### real images
        #if self.opt.isTrain:
        image_path = self.image_paths[index]           
        image = Image.open(image_path).convert('RGB')
        # print(image.width)
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params, normalize=False)      
        image_tensor = transform_image(image)        
        
        # transform reshape the comp_image
        
        # print(image_tensor.shape)
        label_tensor=0
        transform_label = 0
        if not self.opt.no_seg:
                ### label maps
                label_path = self.label_paths[index]
                label = Image.open(label_path)
                                        
                if self.opt.label_nc == 0:
                        transform_label = get_transform(self.opt, params)
                        label_tensor = transform_label(label.convert('RGB'))
                else:                        
                        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                        label_tensor = transform_label(label) * 255.0                                                
                        
        image2 = Image.fromarray(util.tensor2im(image_tensor))
        
        
        ds = 0
        if self.opt.comp_type=='ds':
            ds = image2.resize((image2.size[0]/self.opt.alpha,image2.size[1]/self.opt.alpha), Image.ANTIALIAS)        
            ds = ds.resize((image2.size[0],image2.size[1]), Image.ANTIALIAS)          
            ds_tensor = transform_image(ds)          
                
        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_label(inst)

            if self.opt.load_features:
                raise NotImplementedError("no features")                         

        input_dict = {'label': label_tensor, 'inst': inst_tensor, 'image': image_tensor, 'ds': ds_tensor,
                        'feat': feat_tensor, 'path': image_path}

        return input_dict

