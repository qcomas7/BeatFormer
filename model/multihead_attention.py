import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

"Modified from https://github.com/muqiaoy/dl_signal/blob/master/transformer/modules/multihead_attention.py"


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
        self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
        nn.init.xavier_normal_(self.bias_k)
        nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, i_proj):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        tgt_len, bsz, embed_dim = query.size()
        if qkv_same:
            q, k, v, weight_q, weight_k, weight_v = self.in_proj_qkv(query, i_proj)
        elif kv_same:
            q, weight_q = self.in_proj_q(query, i_proj)
            if key is None:
                k = v = None
            else:
                k, v, weight_k, weight_v = self.in_proj_kv(key, i_proj)
        else:
            q, weight_q = self.in_proj_q(query, i_proj)
            k, weight_k = self.in_proj_k(key, i_proj)
            v, weight_v = self.in_proj_v(value, i_proj)

        q = q*self.scaling
        k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (attn_weights - torch.min(attn_weights)) / (torch.max(attn_weights) - torch.min(attn_weights))
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights, torch.concat([weight_q, weight_k, weight_v], dim=0).unsqueeze(0)

    def in_proj_qkv(self, query, i_proj):
        qkv, w_qkv = self._in_proj(query, i_proj)
        q, k, v = qkv.chunk(3, dim=-1)
        w_q, w_k, w_v = w_qkv.chunk(3, dim=-1)
        return q, k, v, w_q, w_k, w_v

    def in_proj_kv(self, key, i_proj):
        kv, w_kv = self._in_proj(key, i_proj, start=self.embed_dim)
        k, v = kv.chunk(2, dim=-1)
        w_k, w_v = w_kv.chunk(2, dim=-1)
        return k, v, w_k, w_v

    def in_proj_q(self, query, i_proj):
        q, w_q = self._in_proj(query, i_proj, end=self.embed_dim)
        return q, w_q

    def in_proj_k(self, key, i_proj):
        k, w_k = self._in_proj(key, i_proj, start=self.embed_dim, end=2 * self.embed_dim)
        return k, w_k

    def in_proj_v(self, value, i_proj):
        v, w_v = self._in_proj(value, i_proj, start=2 * self.embed_dim)
        return v, w_v

    def _in_proj(self, input, i_proj, start=0, end=None, ):
        weight = self.in_proj_weight.clone()
        bias = self.in_proj_bias
        weight[:, ::3] = i_proj.unsqueeze(-1)
        weight = weight[start:end, :]
        bias = bias[start:end]
        out = F.linear(input, weight, bias)
        return out, weight.T.unsqueeze(0)