import torch
from torch import nn

"""Adapted from the paper: Squared Earth Mover’s Distance Loss for Training Deep Neural Networks on Ordered-Classes """


def earth_mover_distance(y_true, y_pred):
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)


class EMD_Loss(nn.Module):
    def __init__(self, scaling_factor):
        super().__init__()
        self.scaling = scaling_factor

    def forward(self, p, q):
        assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
        mini_batch_size = p.shape[0]
        loss_vector = []
        for i in range(mini_batch_size):
            loss_vector.append(earth_mover_distance(p[i], q[i]))
        return (sum(loss_vector) / mini_batch_size) * self.scaling
