import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, overlap):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = ((img_size - patch_size) // (patch_size-overlap) + 1) ** 2
        self.patch_size = patch_size
        self.overlap = overlap
        self.proj = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size-self.overlap)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = self.proj(x)
        x = x.view(bs, c, self.patch_size, self.patch_size, -1).permute(0, 1, 4, 2, 3)
        return x


