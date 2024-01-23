import os
from .train_options import TrainOptions


class TrainMarsOptions(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
        self.parser.add_argument("--model_type", type=str, default="origin")
        

class ParameterFinenet:
    def __init__(self, ckp_dir, name, batchsize, data_root="datasets/NASA-final", size=1200, no_flip=True,
            finetune_comp=False, is_train=True, test_num=50, continue_train=False,
            random_flip=False, choice="scale_width", scalecrop_ratio=4, scalecropsize=None,
            nonorm=False):
        """_summary_

        Args:
            ckp_dir (_type_): _description_
            name (_type_): _description_
            batchsize (_type_): _description_
            size (int, optional): _description_. Defaults to 1200.
            no_flip (bool, optional): _description_. Defaults to True.
            finetune_comp (bool, optional): _description_. Defaults to False.
            is_train (bool, optional): _description_. Defaults to True.
            test_num (int, optional): _description_. Defaults to 50.
            continue_train (bool, optional): _description_. Defaults to False.
            random_flip (bool, optional): _description_. Defaults to False.
            choice (str, optional): scale_width, scalecrop. Defaults to "scale_width".
            scalecrop_ratio (int, optional): _description_. Defaults to 4.
            scalecropsize (_type_, optional): width, height. Defaults to None.
        """
        self.continue_train = continue_train
        self.dataroot = data_root
        self.loadSize = size
        self.batchSize = batchsize
        self.resize_or_crop = choice
        self.fineSize = 256
        if scalecropsize:
            assert isinstance(scalecropsize, tuple)
            self.fineSizes = scalecropsize
        else:
            self.fineSizes = (1152 // scalecrop_ratio, 1600 // scalecrop_ratio)
        # (bs, ch, h, w)
        # crop and cut the image to finesize
        self.nThreads = 8
        # Thread numbers for loading data
        self.name = name
        self.gpu_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))))
        self.ckp_dir = ckp_dir
        self.norm_fine = "instance"
        self.finetune_comp = finetune_comp
        self.print_freq = 100
        self.train = is_train
        self.isTrain = is_train
        self.no_flip = no_flip
        self.random_flip = random_flip
        self.no_seg = True
        self.ntest = test_num
        self.how_many = test_num
        self.max_dataset_size = test_num
        self.no_instance = True
        self.load_features = False
        self.nonorm = nonorm
        
        if not is_train:
            self.serial_batches = True
        else:
            self.serial_batches = False
            
        self.max_dataset_size = float('inf')
        self.comp_type = "none"
        self.save_dir = os.path.join(ckp_dir, name)
        self.log_dir = os.path.join(self.save_dir, "log")
        if self.train:
            self.log_path = os.path.join(self.log_dir, "train.log")
        else:
            self.log_path = os.path.join(self.log_dir, "test.log")
        
        if is_train:
            self.phase = "train"
        else:
            self.phase = "test"
        
        self.check_dir(self.ckp_dir)
        self.check_dir(self.save_dir)
        self.check_dir(self.log_dir)
        
    def check_dir(self, dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)
    
    def __repr__(self) -> str:
        rep_str = ""
        for k, v in sorted(self.__dict__.items()):
            rep_str += f"{k}\t{v}"
            rep_str += "\n"
            
        return rep_str
        