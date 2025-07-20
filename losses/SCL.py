import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.EMD_loss import EMD_Loss


class SpectralContrastiveLoss(nn.Module):
    def __init__(self, margin, scaling):
        super().__init__()
        self.margin = margin
        self.emd = EMD_Loss(scaling_factor=scaling)

    def forward(self, psds):
        batch_size, num_views, _ = psds.shape
        triu_i, triu_j = torch.triu_indices(num_views, num_views, offset=1)
        pos_pairs = sum(sum(self.emd(psds[b, i], psds[b, j]) for i, j in zip(triu_i, triu_j)) for b in
                        range(batch_size)) / len(triu_i)
        values = torch.arange(0, num_views)
        view_ind = torch.repeat_interleave(values, repeats=num_views)
        neg_pairs = sum(self.emd(psds[0, i], psds[1, j]) for i, j in zip(view_ind, view_ind)) / len(view_ind)
        loss = F.relu(pos_pairs - neg_pairs + self.margin)
        return loss



