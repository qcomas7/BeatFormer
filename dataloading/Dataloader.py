import numpy as np
import time
import csv
import math
import os
import torch
from torch.utils.data import Dataset
import kornia.color as kcolor
from dataloading.utils_data import load_video, RandomPatchOcclusion


class rPPG_Dataloader(Dataset):
    def __init__(self, data_path=None, dataset=None, temporal_length=300, temporal_stride=10, img_size=96,
                 normalize=1, face_mode=3, split_set='Train'):

        start_time = time.time()
        self.dataset = dataset
        self.data_path = data_path
        self.face_mode = face_mode
        self.temporal_length = temporal_length
        self.temporal_stride = temporal_stride
        self.split_set = split_set
        self.img_size = img_size
        self.normalize = normalize
        self.samples = []
        self.videos = []
        self.ppg_gt = []
        self.hr_gt = []
        self.idxs = []

        set_list = []
        dataset_path = os.path.join(self.data_path, self.dataset)
        path = dataset_path + '/Protocols/' + self.split_set + '.txt'
        fdir = open(path, "r")
        for x in fdir:
            dirr = x.split('\n')[0]
            set_list.append(dirr)
        print('------------------------')
        print('Loading ' + self.dataset + ' dataset ...')

        for user in set_list:
            user_path = dataset_path + '/Data/' + user
            bio_file = user_path + '/phys.csv'

            if self.face_mode == 0:  # static
                video_path = user_path + '/vid.avi'
            if self.face_mode == 1:  # static with mask
                video_path = user_path + '/vid_mask.avi'
            if self.face_mode == 2:  # tracking
                video_path = user_path + '/video.avi'
            if self.face_mode == 3:  # tracking with mask
                video_path = user_path + '/video_mask.avi'

            hr_data = []
            ppg_data = []
            time_data = []
            with open(bio_file, 'r') as file:
                bio_reader = csv.reader(file)
                next(bio_reader)
                for item in bio_reader:
                    hr_data.append(item[3])
                    ppg_data.append(item[2])
                    time_data.append(item[1])

            ppg_signal = np.asarray(ppg_data, dtype=np.float32)
            hr_signal = np.asarray(hr_data, dtype=np.float32)
            video_data = np.array(load_video(video_path, type_color='RGB', img_size=self.img_size))
            sub_name = [user for _ in range(len(ppg_signal))]

            print('##############')
            print(user)
            print('##############')
            video_chunks, bvps_chunks, hr_chunks, idx_chunks = self.chunk(video_data, ppg_signal, hr_signal, sub_name,
                                                                          self.temporal_length, self.temporal_stride,
                                                                          split=self.split_set)
            self.videos.extend(video_chunks)
            self.ppg_gt.extend(bvps_chunks)
            self.hr_gt.extend(hr_chunks)
            self.idxs.extend(idx_chunks)
        self.samples = [self.videos, self.ppg_gt, self.hr_gt, self.idxs]
        print("--- %s seconds ---" % (time.time() - start_time))

    def __getitem__(self, sample_idx):
        bvp_gt = torch.from_numpy(self.samples[1][sample_idx])
        hr_gt = torch.from_numpy(self.samples[2][sample_idx])
        idx = self.samples[3][sample_idx]
        video = torch.empty(3, len(self.samples[0][sample_idx]), self.img_size, self.img_size, dtype=torch.float)
        for frame_idx, img in enumerate(self.samples[0][sample_idx]):
            out_type = img.dtype
            out_type_info = np.iinfo(out_type)
            img = torch.from_numpy(img).to(torch.int32).permute(2, 0, 1)
            if self.normalize == 0:
                # Normalize between 0 and 1
                img_range = out_type_info.max - out_type_info.min
                img = (img - out_type_info.min) / img_range
            elif self.normalize == 1:
                # Normalize between 0 and 255
                img = ((img - torch.amin(img)) * 255.0 / (torch.amax(img) - torch.amin(img))).to(torch.int32)
            video[:, frame_idx] = img

        if self.split_set.startswith('Train'):
            video = video.permute(1, 0, 2, 3)
            video_rgb = video.unsqueeze(0)
            video_hsv = self.convert_color_space(video, conversion='hsv').unsqueeze(0)
            video_lab = self.convert_color_space(video, conversion='lab').unsqueeze(0)
            video_flip = torch.flip(video, dims=(0,)).unsqueeze(0)
            video_occ = RandomPatchOcclusion(num_patches=9, occlusion_type='black', p=0.7)(video).unsqueeze(0)
            return torch.concat([video_rgb, video_hsv, video_lab, video_flip, video_occ], dim=0), bvp_gt, hr_gt, idx
        else:
            return video, bvp_gt, hr_gt, idx

    @staticmethod
    def chunk(frames, bvps, hrs, sub_name, temporal_length, temporal_overlap, split):
        if split.startswith('Test') or split == 'Dev':
            clip_num = math.ceil((frames.shape[0] - temporal_length) / (temporal_length - temporal_overlap)) + 1
        else:
            clip_num = int((frames.shape[0] - temporal_length) / (temporal_length - temporal_overlap)) + 1
        index = (temporal_length - temporal_overlap)
        frames_clips = [frames[i * index:i * index + temporal_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * index:i * index + temporal_length] for i in range(clip_num)]
        hrs_clips = [hrs[i * index:i * index + temporal_length] for i in range(clip_num)]
        name_clips = [sub_name[i * index:i * index + temporal_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips), np.array(hrs_clips), name_clips

    @staticmethod
    def convert_color_space(input_tensor, conversion='lab'):
        if conversion.lower() == 'lab':
            x_conv = kcolor.rgb_to_lab(input_tensor)
        elif conversion.lower() == 'hsv':
            x_conv = kcolor.rgb_to_hsv(input_tensor)
        elif conversion.lower() == 'yuv':
            x_conv = kcolor.rgb_to_yuv(input_tensor)
        else:
            raise ValueError(f"Unsupported conversion: {conversion}")
        return x_conv

    def __len__(self):
        return len(self.samples[0])