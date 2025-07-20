from model.CZT_model import *
from model.Patch_embedding import PatchEmbedding
from model.ZOCA import ZOCA, ComplexLinear
from ptflops import get_model_complexity_info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BeatFormer(nn.Module):
    def __init__(self, in_chans: int = 3, seq: int = 300, window_size: int = 60, img_size: int = 96, patches: int = 16, overlap=0,
                 heads: int = 3, layers: int = 4, dropout: float = 0., fs: int = 30):
        super().__init__()
        self.in_chans = in_chans
        self.seq = seq
        self.window_size = window_size
        self.heads = heads
        self.layers = layers
        self.bin_res = ((2.5 - 0.66) / self.window_size)
        self.dropout = dropout
        self.device = device
        self.fs = fs
        self.patch_embedding = PatchEmbedding(img_size, patches, overlap)
        self.patches = self.patch_embedding.n_patches
        self.czt = CZT(input_size=self.window_size, fs=self.fs, bin_res=self.bin_res, device=self.device)
        self.i_proj = nn.Parameter(torch.ones(1, self.in_chans*self.patches), requires_grad=False).to(device)
        self.pe = nn.Parameter(torch.zeros((self.window_size, (self.seq - self.window_size) + 1, 1)), requires_grad=True)
        self.ZOCA_blocks = torch.nn.ModuleList([ZOCA(q_dim=self.in_chans * self.patches, n_heads=self.heads,
                                                     attn_pdrop=self.dropout) for _ in range(self.layers)])
        self.fc = ComplexLinear(self.in_chans * self.patches, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        B, seq, C,  W, H = x.shape
        x = x.reshape(B * seq, C, H, W)
        x = self.patch_embedding(x).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, seq, *x.shape[1:])
        y = torch.mean(x, dim=[3, 4])

        windows = y.unfold(1, self.window_size, 1)
        windows = windows.reshape(windows.shape[0]*windows.shape[1], self.in_chans, self.window_size)
        windows = windows.permute(2, 0, 1)
        windows = torch.divide(windows, torch.mean(windows, dim=0)) - 1
        windows[torch.isnan(windows)] = 0

        _, X = self.czt(windows)
        X_split = X.chunk(2, dim=0)
        X_re = (X_split[0] + self.pe.repeat(1, B, self.in_chans * self.patches)) / windows.shape[0]
        X_im = (X_split[1] + self.pe.repeat(1, B, self.in_chans * self.patches)) / windows.shape[0]

        qkv_weights = []
        X_re_pred = X_re
        X_im_pred = X_im
        for sa_layer in self.ZOCA_blocks:
            X_re_pred, X_im_pred, weights = sa_layer(X_re_pred, X_im_pred, self.i_proj)
            qkv_weights.append(weights)
        qkv_weights = torch.stack(qkv_weights)
        X_re_pred, X_im_pred = self.fc(X_re_pred, X_im_pred)
        P_ = self.czt.reverse(torch.concat([X_re_pred, X_im_pred], dim=0), False).flatten(1)
        P_split = P_.chunk(B, dim=-1)
        P = torch.stack(P_split)
        P = (P - torch.mean(P, dim=1, keepdim=True)) / (torch.std(P, dim=1, keepdim=True) + 1e-8)
        P[torch.isnan(P)] = 0
        pulse_signal = torch.zeros(B, seq, device=self.device)
        for b in range(B):
            for i in range(P.shape[-1]):
                pulse_signal[b, i:i + self.window_size] += P[b, :, i]
        return pulse_signal, qkv_weights


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BeatFormer(3, 300, 60, 96, 96, 0, 3, 4, 0, 30).to(device)
    input_shape = (300, 3, 96, 96)
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)
    print(f"GMACs: {macs}")
    print(f"Parameters: {params}")

# if __name__ == '__main__':
#      main()
