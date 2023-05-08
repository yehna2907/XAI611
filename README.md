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

#### Download Datasets
Download Kinetics-400 and UCF101. Please refer to [MVFNet](https://github.com/whwu95/MVFNet/blob/main/data_process/DATASETS.md) repo for the detailed guide of data processing.  
The annotation file is a text file with multiple lines, and each line indicates the directory to frames of a video, total frames of the video and the label of a video, which are split with a whitespace. Here is the format: 
```sh
abseiling/-7kbO0v4hag_000107_000117 300 0
abseiling/-bwYZwnwb8E_000013_000023 300 0
```


## Pre-train on Kinetics-400
```sh
# For example, train the 8 Frames ViT-B/32.
sh scripts/run_train.sh  configs/k400/k400_train_video_vitb-32-f8.yaml
```

If you're using multiple GPUs, edit the number of `--nproc_per_node` in script file.
```sh
# For example, if you are training with 8 GPUs do, modify it as follows:
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