import math
import torch
from torch.autograd import Variable
import numpy as np
from torch import nn

""" Modified from https://github.com/radimspetlik/hr-cnn """


class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)
        k = k.type(torch.FloatTensor).cuda()
        two_pi_n_over_N = two_pi_n_over_N.cuda()
        hanning = hanning.cuda()
        output = output * hanning
        output = output.unsqueeze(1)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        sin_component = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1)
        cos_component = torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1)
        complex_absolute = sin_component ** 2 + cos_component ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        batch_size, N = output.size()
        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)
        complex_absolute = (1.0 / complex_absolute.sum(dim=1, keepdim=True)) * complex_absolute
        return complex_absolute


def PSD_estimation(inputs, Fs):
    softmax = nn.Softmax(dim=1)
    bpm_range = torch.arange(40, 150, dtype=torch.float).cuda()
    complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
    complex_absolute = softmax(complex_absolute)
    return complex_absolute


