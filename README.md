## Prerequisites
- PyTorch >= 1.8
- torchvision
- RandAugment
- pprint
- tqdm
- dotmap
- yaml
- csv


## Data Preparation

#### Download and Preprocessing
Download Kinetics-400 and UCF101. Please refer to [MVFNet](https://github.com/whwu95/MVFNet/blob/main/data_process/DATASETS.md) repo for downloading the datasets.  
Make sure that the videos are under the class directory. For example:
```sh
train/abseiling/-7kbO0v4hag_000107_000117.mp4
train/abseiling/-bwYZwnwb8E_000013_000023.mp4
```

After that, the videos should be resized to 320. The code for resizing can also be found here: [MVFNet](https://github.com/whwu95/MVFNet/blob/main/data_process/DATASETS.md).


## Pre-train on Kinetics-400
```sh
# For example, train the 8 Frames ViT-B/32.
sh scripts/run_train.sh  configs/k400/k400_train_video_vitb-32-f8.yaml
```

If you're using multiple GPUs, edit the number of `--nproc_per_node` in script file.
```sh
# For example, if you are training with 4 GPUs do, modify it as follows:
python -m torch.distributed.launch --master_port 1237 --nproc_per_node=4 \
         train.py  --config ${config} --log_time $now
```


## Fine-tune on UCF101
```sh
sh scripts/run_train.sh  configs/ucf101/ucf_k400_finetune.yaml
```


## Test
```sh
sh scripts/run_test.sh  configs/ucf101/ucf_k400_finetune.yaml exp/k400/ViT-B/32/f8/last_model.pt
```