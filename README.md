
### Environments
```sh
conda env create -f environment.yml
pip install -e .
```

### Pretrained models
link https://pan.baidu.com/s/1Tov1WCrZnq0qpCz-ofxhWQ?pwd=hh4h with extraction code `hh4h` 
comming soon

### datasets
comming soon

### Get the compression results
comming soon
```sh

```

### First train a structure extraction network

```sh
cd get_pretrain_model/A_DSSLIC_simple
python get_pretrain_model/A_DSSLIC_simple/train_finenet_fast.py -d $pathToDataset
```

### Second train our compression model based on the structure extraction network

- Get the psnr model
```sh
CUDA_VISIBLE_DEVICES=0 python examples/train.py -m model_cheng_anchor_win-attn5 -d $pathToDataset --batch-size 16 -lr 1e-4 --save --cuda --epoch 1000 --patch-size 256 256 --lambda 0.01 --test-batch-size 1
```

- Get the ms-ssim model
```sh
CUDA_VISIBLE_DEVICES=0 python examples/train.py -m model_cheng_anchor_win-attn5 -d $pathToDataset --batch-size 16 -lr 3e-4 --save --cuda --epoch 1000 --patch-size 256 256 --lambda 0.01 --test-batch-size 1 --metric ms_ssim
```

