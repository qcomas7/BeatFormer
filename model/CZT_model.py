import torch
from torch import nn
import numpy as np


def W_weights(freq_zoom, fs, N, grad, device, reverse=False):
    M = len(freq_zoom)
    t = (torch.arange(N) / fs).to(device)
    dt = (t[-1] - t[0]) / (len(t) - 1)
    df = (freq_zoom[-1] - freq_zoom[0]) / (M - 1)
    k_vals, n_vals = np.mgrid[0:M, 0:N]
    k_vals = torch.from_numpy(k_vals).to(device)
    n_vals = torch.from_numpy(n_vals).to(device)
    if reverse:
        W_vals = (2 * torch.pi * df * k_vals * n_vals * dt)
    else:
        W_vals = (-2 * torch.pi * df * k_vals * n_vals * dt)
    W_real = nn.Parameter(torch.cos(W_vals), requires_grad=grad).to(device)
    W_imag = nn.Parameter(torch.sin(W_vals), requires_grad=grad).to(device)
    return W_real, W_imag


class A_weights(torch.nn.Module):
    def __init__(self, freq_zoom, fs, N, grad, device):
        super().__init__()
        self.fs = fs
        self.N = N
        self.device = device
        self.freq_zoom = freq_zoom
        self.freq_init = torch.nn.Parameter(self.freq_zoom[0], requires_grad=grad)

    def A_real_function(self):
        t = (torch.arange(self.N) / self.fs).to(self.device)
        dt = ((t[-1] - t[0]) / (len(t) - 1)).to(self.device)
        k = torch.arange(self.N).to(self.device)
        A_vals = (2 * torch.pi * self.freq_init * -k * dt).to(self.device)
        return torch.diag(torch.cos(A_vals)).to(self.device)

    def A_imag_function(self):
        t = (torch.arange(self.N) / self.fs).to(self.device)
        dt = ((t[-1] - t[0]) / (len(t) - 1)).to(self.device)
        k = torch.arange(self.N).to(self.device)
        A_vals = (2 * torch.pi * self.freq_init * -k * dt).to(self.device)
        return torch.diag(torch.sin(A_vals)).to(self.device)

    def forward(self, x):
        A_real = self.A_real_function()
        Ax_real = torch.einsum('ji,ics->jcs', A_real, x)
        A_imag = self.A_imag_function()
        Ax_imag = torch.einsum('ji,ics->jcs', A_imag, x)
        return Ax_real, Ax_imag

    def reverse(self, x_f):
        M = len(self.freq_zoom)
        t = (torch.arange(M) / self.fs).to(self.device)
        df = (self.freq_zoom[-1] - self.freq_init) / (M - 1)
        k = torch.arange(M).to(self.device)
        A_vals = (2 * torch.pi * df * k * t[0])
        A_real = torch.diag(torch.cos(A_vals))
        A_img = torch.diag(torch.sin(A_vals))
        x_f_split = x_f.chunk(2, dim=0)
        x_real = x_f_split[0]
        x_imag = x_f_split[-1]
        Ax_real = torch.einsum('ji,ics->jcs', A_real, x_real) - torch.einsum('ji,ics->jcs', A_img, x_imag)
        Ax_img = torch.einsum('ji,ics->jcs', A_real, x_imag) - torch.einsum('ji,ics->jcs', A_img, x_real)
        return Ax_real, Ax_img


class Hamming_Window(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.alpha = torch.nn.Parameter(torch.tensor(0.54), requires_grad=True).to(self.device)
        self.beta = torch.nn.Parameter(torch.tensor(0.46), requires_grad=True).to(self.device)
        self.filtered_sig = []

    def hamming_function(self, data):
        window = self.alpha - self.beta * torch.cos(
            torch.pi * 2 * torch.linspace(0, data.shape[0], data.shape[0]) / data.shape[0]).to(self.device)
        data = window * data
        return data

    def forward(self, data):
        shape = data.shape
        for i in range(0, shape[0]):
            filter_batch = self.hamming_function(data[i, :])
            self.filtered_sig.append(filter_batch.unsqueeze(0))
        signal_batch = torch.cat(self.filtered_sig, 0)
        self.filtered_sig = []
        return signal_batch


class CZT(nn.Module):
    def __init__(self, input_size, fs, bin_res, device):
        super(CZT, self).__init__()

        self.input_size = input_size
        self.fs = fs
        self.device = device
        self.freq_zoom = torch.arange(0.66, 2.5, bin_res)
        self.A_fc = A_weights(self.freq_zoom, fs, self.input_size, False, self.device)
        self.W_real, self.W_imag = W_weights(freq_zoom=self.freq_zoom, fs=self.fs, N=self.input_size, grad=False,
                                             device=self.device, reverse=False)
        self.iW_real, self.iW_imag = W_weights(freq_zoom=self.freq_zoom, fs=self.fs, N=len(self.freq_zoom), grad=False,
                                               device=self.device, reverse=True)
        self.hardtanh = nn.Hardtanh()

    def forward(self, x):
        Ax_real, Ax_imag = self.A_fc(x)
        Ax = torch.vstack([Ax_real, Ax_imag])
        W = torch.vstack([torch.hstack([self.W_real, -self.W_imag]), torch.hstack([self.W_imag, self.W_real])])
        W = self.hardtanh(W)
        X = torch.einsum('ji,ics->jcs', W, Ax)
        return W, X

    def reverse(self, x, diff=False):
        if diff:
            x = self.freq_diff(x)
        t = (torch.arange(len(self.freq_zoom)) / self.fs).to(self.device)
        phase_vals = (2 * torch.pi * self.freq_zoom[0] * t)
        phase_real = torch.diag(torch.cos(phase_vals))
        phase_img = torch.diag(torch.sin(phase_vals))
        Ax_f_real, Ax_f_imag = self.A_fc.reverse(x)
        Ax_f = torch.vstack([Ax_f_real, Ax_f_imag])
        iW = torch.vstack([torch.hstack([self.iW_real, -self.iW_imag]), torch.hstack([self.iW_imag, self.iW_real])])
        iW = self.hardtanh(iW)
        xt = torch.einsum('ji,ics->jcs', iW, Ax_f)
        xt_split = xt.chunk(2, dim=0)
        xt_real = xt_split[0]
        xt_imag = xt_split[-1]
        real = torch.einsum('ji,ics->jcs', phase_real, xt_real) - torch.einsum('ji,ics->jcs', phase_img, xt_imag)
        return real / len(self.freq_zoom)

    def freq_diff(self, x_f):
        der_vals = ((2 * torch.pi) * self.freq_zoom).to(self.device)
        der_freq = torch.diag(der_vals)
        x_f_split = x_f.chunk(2, dim=0)
        x_f_real = x_f_split[0]
        x_f_imag = x_f_split[-1]
        dX_r = x_f_real - torch.einsum('ji,ics->jcs', der_freq, x_f_imag)
        dX_i = x_f_imag + torch.einsum('ji,ics->jcs', der_freq, x_f_real)
        return torch.vstack([dX_r, dX_i])