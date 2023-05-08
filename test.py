import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms._transforms_video as transforms_video
import time
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
import clip

import yaml
from dotmap import DotMap

from datasets.datasets import VideoDataset
from modules.video_clip import video_header, VideoCLIP


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='global config file')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )                        
    parser.add_argument('--test_crops', type=int, default=1)   
    parser.add_argument('--test_clips', type=int, default=1) 
    parser.add_argument('--dense', default=False, action="store_true",
                    help='use multiple clips for test')                     
    args = parser.parse_args()
    return args


def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict


def main(args):
    init_distributed_mode(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    # get fp16 model and weight
    model, clip_state_dict = clip.load(
        config.network.arch,
        device='cpu', jit=False,
        internal_modeling=config.network.tm,
        T=config.data.frames_per_clip,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st= config.network.joint_st) # Must set jit=False for training  ViT-B/32

    # video augmentation
    video_augmentation = torchvision.transforms.Compose([
            transforms_video.ToTensorVideo(),
            transforms_video.RandomResizedCropVideo(config.data.input_size, (0.2, 1))
    ])

    video_head = video_header(
        config.network.sim_header,
        clip_state_dict)

    if args.precision == "amp" or args.precision == "fp32":
        model = model.float()

    val_data = VideoDataset(
        config.data.val_root,
        config.data.dataset,
        config.data.label_list,
        config.data.frames_per_clip,
        num_workers=config.data.frames_per_clip,
        transform=video_augmentation
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=val_sampler, pin_memory=True, drop_last=False)

    model_full = VideoCLIP(model, video_head, config.data.frames_per_clip)

    if os.path.isfile(args.weights):
        checkpoint = torch.load(args.weights, map_location='cpu')
        if dist.get_rank() == 0:
            print('load model: epoch {}'.format(checkpoint['epoch']))

        model_full.load_state_dict(update_dict(checkpoint['model_state_dict']))
        del checkpoint

    if args.distributed:
        model_full = DistributedDataParallel(model_full.cuda(), device_ids=[args.gpu], find_unused_parameters=True)

    classes = torch.cat([clip.tokenize(c) for i, c in enumerate(val_data.classes)])

    #### generate classes feature ######
    class_feats_file = 'text_feats_{}_{}.pt'.format(config['data']['dataset'], config['network']['arch']).replace('/', '')
    if os.path.isfile(class_feats_file):
        print('=> load classes features from {}'.format(class_feats_file))
        classes_features = torch.load(class_feats_file)
    else:
        model.eval()
        with torch.no_grad():
            classes_features = model.encode_text(classes)
        # if dist.get_rank() == 0:
        #     torch.save(classes_features.cpu(), class_feats_file)

    prec1 = validate(
        val_loader, device, 
        model_full, config, classes_features, args.test_crops, args.test_clips)

    return


def validate(val_loader, device, model, config, text_features, test_crops, test_clips):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    proc_start_time = time.time()

    with torch.no_grad():
        n_class = text_features.size(0)
        
        for i, (video, class_id) in enumerate(val_loader):
            batch_size = class_id.numel()
            num_crop = test_crops
            num_crop *= test_clips  # 4 clips for testing when using dense sample

            b, c, t, h, w = video.size()
            class_id = class_id.to(device)
            text_features = text_features.to(device)
            video_input = video.to(device).view(-1, c, h, w)

            video_embedding = model.module.encode_image(video_input)
            cnt_time = time.time() - proc_start_time
            video_embedding = video_embedding.reshape(batch_size, num_crop, -1).mean(1)  # bs dim

            video_embedding /= video_embedding.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * video_embedding @ text_features.T)
            similarity = similarity.view(batch_size, -1, n_class).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)      # bs 200

            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))
    
            if i % config.logging.print_freq == 0 and dist.get_rank() == 0:
                runtime = float(cnt_time) / (i+1) / (batch_size * dist.get_world_size())
                print(
                    ('Test: [{0}/{1}], average {runtime:.4f} sec/video \t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), runtime=runtime, top1=top1, top5=top5)))

    if dist.get_rank() == 0:
        print('-----Evaluation is finished------')
        print('Overall Prec@1 {:.03f}% Prec@5 {:.03f}%'.format(top1.avg, top5.avg))
    
    return top1.avg


if __name__ == '__main__':
    args = get_parser()
    main(args)
