
### Environments

Use the environment.yml file to create the python environment

```sh
conda env create -f environment.yml
pip install -e .
```

### Pretrained models
The pretrained model is available on https://pan.baidu.com/s/1Tov1WCrZnq0qpCz-ofxhWQ?pwd=hh4h with extraction code `hh4h`. We provided both the cheng model and our LSH model. 

### datasets

We used the dataset according to the article https://ieeexplore.ieee.org/abstract/document/10008891. The dataset is comming soon on https://github.com/dq0309/MIC-Dataset.

### Get the compression results

comming soon

### First train a structure extraction network

```sh
cd get_pretrain_model/A_DSSLIC_simple
python get_pretrain_model/A_DSSLIC_simple/train_finenet_fast.py -d $pathToDataset
```

### Second train our compression model based on the structure extraction network
- put the weight of the structure extraction network in `pretrain_models` folder or download the pretrained_compnet.pth from the link above and put it to 
the corresponding directory
```sh
mkdir pretrain_models
mv $your_model.pth  pretrain_models/pretrained_compnet.pth
```

- Get the psnr model
```sh
CUDA_VISIBLE_DEVICES=0 python examples/train.py -m model_cheng_anchor_win-attn5 -d $pathToDataset --batch-size 16 -lr 1e-4 --save --cuda --epoch 1000 --patch-size 256 256 --lambda 0.01 --test-batch-size 1
```

- Get the ms-ssim model
```sh
CUDA_VISIBLE_DEVICES=0 python examples/train.py -m model_cheng_anchor_win-attn5 -d $pathToDataset --batch-size 16 -lr 3e-4 --save --cuda --epoch 1000 --patch-size 256 256 --lambda 0.01 --test-batch-size 1 --metric ms_ssim
```

