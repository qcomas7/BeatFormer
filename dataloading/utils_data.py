import torch
import os
import cv2
import random
random.seed(42)


def check_file_exist(path: str):
    if not os.path.exists(path):
        raise Exception('Can not find path: "{}"'.format(path))


def load_video(path: str, type_color: str, img_size: int) -> list:
    check_file_exist(path)
    video = cv2.VideoCapture(path)
    frames = []
    ret_val, frame = video.read()
    while ret_val:
        if type_color == 'BGR':
            frames.append(cv2.resize(frame, (img_size, img_size), interpolation=0))
        elif type_color == 'RGB':
            frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_size, img_size), interpolation=0))
        ret_val, frame = video.read()
    video.release()
    return frames


class RandomPatchOcclusion:
    def __init__(self, num_patches=9, occlusion_type='noise', p=0.5):
        self.grid_size = int(num_patches ** 0.5)
        self.patch_size = 96 // self.grid_size
        self.occlusion_type = occlusion_type
        self.p = p

    def __call__(self, video):
        if random.random() > self.p:
            return video
        i = random.randint(0, self.grid_size - 1)
        j = random.randint(0, self.grid_size - 1)
        y_start = i * self.patch_size
        y_end = (i + 1) * self.patch_size
        x_start = j * self.patch_size
        x_end = (j + 1) * self.patch_size
        return self._occlude_region(video, y_start, y_end, x_start, x_end)

    def _occlude_region(self, video, y_start, y_end, x_start, x_end):
        occluded = video.clone()
        patch = video[:, :, y_start:y_end, x_start:x_end]
        if self.occlusion_type == 'noise':
            noise = torch.randn_like(patch) * patch.std() + patch.mean()
            occluded[:, :, y_start:y_end, x_start:x_end] = noise
        elif self.occlusion_type == 'black':
            occluded[:, :, y_start:y_end, x_start:x_end] = 0
        elif self.occlusion_type == 'mean':
            channel_means = video.mean(dim=(0, 2, 3), keepdim=True)
            occluded[:, :, y_start:y_end, x_start:x_end] = channel_means
        elif self.occlusion_type == 'random':
            return self._random_occlusion(video, y_start, y_end, x_start, x_end)
        return occluded

    def _random_occlusion(self, video, y_start, y_end, x_start, x_end):
        types = ['noise', 'black', 'mean']
        selected_type = random.choice(types)
        self.occlusion_type = selected_type
        return self._occlude_region(video, y_start, y_end, x_start, x_end)