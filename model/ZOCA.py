import torch
from torch import nn
from model.multihead_attention import MultiheadAttention


class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input_r, input_i):
        return self.fc_r(input_r) - self.fc_i(input_i), self.fc_r(input_i) + self.fc_i(input_r)


class ZOCA(torch.nn.Module):
    def __init__(self, q_dim, n_heads=1, attn_pdrop=0.):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim=q_dim, num_heads=n_heads, attn_dropout=attn_pdrop)
        self.fc1 = ComplexLinear(q_dim, q_dim)
        self.fc2 = ComplexLinear(q_dim, 1)
        self.gelu = nn.GELU()
        self.norm1 = nn.LayerNorm(q_dim)
        self.norm2 = nn.LayerNorm(q_dim)
        self.q_dim = q_dim

    def forward(self, real, imag, i_proj):
        residual_re = real
        residual_im = imag

        i_orthonormal = (i_proj / torch.norm(i_proj[:, :3], dim=1)).repeat(1, 3)
        x_rrr, att_rrr, weights_rrr = self.attn(real, real, real, i_orthonormal)
        x_rri, att_rri, weights_rri = self.attn(real, real, imag, i_orthonormal)
        x_rir, att_rir, weights_rir = self.attn(real, imag, real, i_orthonormal)
        x_irr, att_irr, weights_irr = self.attn(imag, real, real, i_orthonormal)
        x_rii, att_rii, weights_rii = self.attn(real, imag, imag, i_orthonormal)
        x_iri, att_iri, weights_iri = self.attn(imag, real, imag, i_orthonormal)
        x_iir, att_iir, weights_iir = self.attn(imag, imag, real, i_orthonormal)
        x_iii, att_iii, weights_iii = self.attn(imag, imag, imag, i_orthonormal)

        X_re = x_rrr - x_rii - x_iri - x_iir
        X_im = -x_iii + x_irr + x_rir + x_rri

        X_re, X_im = self.fc1(X_re, X_im)
        X_re = self.gelu(X_re)
        X_im = self.gelu(X_im)
        S_re, S_im = self.fc2(X_re, X_im)

        S_re = torch.mul(S_re, S_re) - torch.mul(S_im, -S_im)
        S_im = torch.mul(S_re, -S_im) + torch.mul(S_im, -S_im)
        F_re_t = torch.sum(torch.mul(residual_re, residual_re) - torch.mul(residual_im, -residual_im),
                           dim=-1).unsqueeze(-1)
        F_im_t = torch.sum(torch.mul(residual_re, -residual_im) + torch.mul(residual_im, -residual_im),
                           dim=-1).unsqueeze(-1)
        W_re = torch.div(torch.mul(S_re, F_re_t) + torch.mul(S_im, F_im_t), F_re_t ** 2 + F_im_t ** 2)
        W_im = torch.div(torch.mul(S_re, F_re_t) + torch.mul(S_im, F_im_t), F_re_t ** 2 + F_im_t ** 2)

        W_real = torch.tile(W_re, [1, 1, self.q_dim])
        W_imag = torch.tile(W_im, [1, 1, self.q_dim])

        X_r = torch.mul(W_real, residual_re) - torch.mul(W_imag, residual_im)
        X_i = torch.mul(W_real, residual_im) + torch.mul(W_imag, residual_im)

        X_r = self.norm1(X_r)
        X_i = self.norm1(X_i)

        weights = torch.concat([weights_rrr, weights_rri, weights_rir, weights_irr, weights_rii, weights_iri,
                                weights_iir, weights_iii], dim=0)
        return X_r, X_i, weights