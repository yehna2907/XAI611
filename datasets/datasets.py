import os
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torchvision.io import read_video
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset
from typing import Any, Dict, Union
from tqdm import tqdm
from numpy.random import randint
from torchvision.datasets.video_utils import _VideoTimestampsDataset
import warnings

warnings.filterwarnings("ignore", message="The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")


class VideoDataset(Dataset):
    def __init__(self, root, dataset, labels_file, frames_per_clip,
                 extensions=('mp4',), transform=None, num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, reading_method=0):
        super().__init__()

        self.class_file = pd.read_csv(labels_file, index_col=1, squeeze=True)
        class_to_idx = self.class_file.to_dict()
        self.frames_per_clip = frames_per_clip

        if 'mp4' not in extensions:
            root = self.convert_to_mp4(root, class_to_idx)
            extensions = ('mp4')

        self.root = root
        split = root.split('/')[-1].strip('/')
        metadata_filepath = os.path.join(root, '{}_metadata_{}.pt'.format(dataset, split))

        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.video_paths = [x[0] for x in self.samples]
        self.class_list = [x[1] for x in self.samples]

        if os.path.exists(metadata_filepath):
            self.metadata = torch.load(metadata_filepath)
            self._init_from_metadata(self.metadata)
        else:
            self.metadata = self._compute_frame_pts(num_workers)
            torch.save(self.metadata, metadata_filepath)

        self.transform = transform
        self.reading_method = reading_method

    @property
    def classes(self):
        return self.class_file.keys().tolist()

    def _init_from_metadata(self, metadata: Dict[str, Any]) -> None:
        self.video_paths = metadata["video_paths"]
        assert len(self.video_paths) == len(metadata["video_pts"])
        self.video_pts = metadata["video_pts"]
        assert len(self.video_paths) == len(metadata["video_fps"])
        self.video_fps = metadata["video_fps"]

    def convert_to_mp4(self, directory, class_to_idx):
        path, dir = os.path.split(directory)
        new_dir = os.path.join(path, '{}_mp4'.format(dir))
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)

        for target_class in sorted(class_to_idx.keys()):
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue

            new_cls_dir = os.path.join(new_dir, target_class)
            if not os.path.isdir(new_cls_dir):
                os.mkdir(new_cls_dir)

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    file_path = os.path.join(root, fname)
                    if os.path.isfile(file_path):
                        fname_only, ext = fname.split('.')
                        dest_file = os.path.join(new_cls_dir, fname_only + '.mp4')
                        if ext != 'mp4' and not os.path.isfile(dest_file):
                            os.system('ffmpeg -i {} -c:a copy {}'.format(file_path, dest_file))
        return new_dir

    def _compute_frame_pts(self, num_workers) -> dict[str, Union[Union[list[Any], list[Tensor]], Any]]:
        self.video_pts = []
        self.video_fps = []

        import torch.utils.data

        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(self.video_paths),  # type: ignore[arg-type]
            batch_size=16,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )

        with tqdm(total=len(dl)) as pbar:
            for batch in dl:
                pbar.update(1)
                clips, fps = list(zip(*batch))
                clips = [torch.as_tensor(c, dtype=torch.long) for c in clips]
                self.video_pts.extend(clips)
                self.video_fps.extend(fps)

        return {
            'video_paths': self.video_paths,
            'video_pts': self.video_pts,
            'video_fps': self.video_fps,
        }

    def _uniform_sample(self, video_pts, seg_length=1):
        if len(video_pts) <= self.frames_per_clip:
            offsets = np.concatenate((
                np.arange(len(video_pts)),
                randint(len(video_pts),
                size=self.frames_per_clip - len(video_pts))))
            return video_pts[offsets], np.array(offsets)

        total_frames = len(video_pts)
        ticks = np.linspace(0, total_frames - 1, self.frames_per_clip + 1)
        ticks = np.round(ticks).astype(np.int64)

        offsets = list()
        for i in range(self.frames_per_clip):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= seg_length:
                tick += randint(tick_len - seg_length + 1)
            offsets.extend([j for j in range(tick, tick + seg_length)])

        return video_pts[offsets], np.array(offsets)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_pts = self.video_pts[idx]
        sample_pts, indices = self._uniform_sample(video_pts)

        if self.reading_method == 0:    # 1. Load pts[0] ~ pts[-1] 2. Select Frame
            loaded_frames = read_video(
                video_path,
                sample_pts[0].item(),
                sample_pts[-1].item(),
                pts_unit='pts',
            )[0]

            indices = indices - indices[0]
            indices[indices >= len(loaded_frames)] = len(loaded_frames) - 1
            video = loaded_frames[indices]
        elif self.reading_method == 1:    # load every pts

            video = torch.stack([
                read_video(video_path, pts.item(), pts.item(), pts_unit='pts')[0][0] for pts in sample_pts
            ], dim=0)

        else:
            assert NotImplementedError

        video = self.transform(video)

        return video, self.class_list[idx]

    def __len__(self):
        return len(self.video_paths)
