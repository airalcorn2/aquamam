# Simplified PyTorch Transformer layers/functions when using mostly the defaults.

import torch
import torch.nn.functional as F

from torch.nn import Dropout, LayerNorm, Linear, Module, Parameter
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.modules.activation import NonDynamicallyQuantizableLinear
from torch.nn.modules.transformer import _get_clones


def multi_head_attention_forward(
    query,
    key,
    value,
    in_proj_weight,
    in_proj_bias,
    num_heads,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=True,
    attn_mask=None,
    mem_kv=None,
):
    (q, k, v) = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    (tgt_len, bsz, embed_dim) = query.shape
    head_dim = embed_dim // num_heads
    # (N * num_heads, S, head_dim)
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    if mem_kv is not None:
        (mem_k, mem_v) = mem_kv
        k = torch.cat([mem_k, k], dim=1)
        v = torch.cat([mem_v, v], dim=1)

    if not training:
        dropout_p = 0.0

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)

    attn_output = F._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)[0]
    attn_output = attn_output.transpose(0, 1).contiguous()
    attn_output = attn_output.view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    return (attn_output, (k, v))


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.register_parameter("q_proj_weight", None)
        self.register_parameter("k_proj_weight", None)
        self.register_parameter("v_proj_weight", None)
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value, attn_mask=None, mem_kv=None):
        return multi_head_attention_forward(
            query,
            key,
            value,
            self.in_proj_weight,
            self.in_proj_bias,
            self.num_heads,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            attn_mask,
            mem_kv,
        )


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = activation

    def forward(self, src, src_mask=None, mem_kv=None, return_kv=False):
        x = src
        if self.norm_first:
            (out, out_kv) = self._sa_block(self.norm1(x), src_mask, mem_kv)
            x = x + out
            x = x + self._ff_block(self.norm2(x))
        else:
            (out, out_kv) = self._sa_block(x, src_mask, mem_kv)
            x = self.norm1(x + out)
            x = self.norm2(x + self._ff_block(x))

        if return_kv:
            return (x, out_kv)
        else:
            return x

    # self-attention block
    def _sa_block(self, x, attn_mask, mem_kv):
        (out, out_kv) = self.self_attn(x, x, x, attn_mask=attn_mask, mem_kv=mem_kv)
        return (self.dropout1(out), out_kv)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask=None, mem_kv=None, return_kv=False):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, mem_kv=mem_kv, return_kv=return_kv)

        return output


def main():
    from torch import nn

    d_model = 512
    nhead = 8
    dropout = 0.1
    dim_feedforward = 4 * d_model
    encoder_layer = nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, F.gelu, 1e-6, norm_first=True
    )
    num_layers = 6
    og_transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    og_transformer.eval()

    test_f = "test.pth"
    torch.save(og_transformer.state_dict(), test_f)

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, F.gelu, 1e-6, norm_first=True
    )
    transformer = TransformerEncoder(encoder_layer, num_layers)
    transformer.load_state_dict(torch.load(test_f))
    transformer.eval()

    (S, N) = (16, 32)
    x = torch.rand(S, N, d_model)

    # No mask.
    with torch.no_grad():
        og_out = og_transformer(x)
        out = transformer(x)

    assert torch.allclose(og_out, out)

    # With mask.
    mask = torch.triu(torch.ones(S, S) * float("-inf"), diagonal=1)
    n_preds = 3
    mask[:, : -(n_preds - 1)] = 0
    with torch.no_grad():
        og_out = og_transformer(x, mask)
        out = transformer(x, mask)

    assert torch.allclose(og_out, out)

    # Cached.
    mem_kvs = []
    first_xs = x[: -(n_preds - 1)]
    for mod in transformer.layers:
        with torch.no_grad():
            (first_xs, mem_kv) = mod(first_xs, return_kv=True)

        mem_kvs.append(mem_kv)

    assert torch.allclose(out[: -(n_preds - 1)], first_xs, atol=5e-6)

    last_xs = []
    for which_val in range(n_preds - 1):
        idx = len(first_xs) + which_val
        next_x = x[idx : idx + 1]
        for (mod_idx, mod) in enumerate(transformer.layers):
            with torch.no_grad():
                (next_x, mem_kv) = mod(next_x, mem_kv=mem_kvs[mod_idx], return_kv=True)

            mem_kvs[mod_idx] = mem_kv

        last_xs.append(next_x)

    last_xs = torch.cat(last_xs)
    # Not sure why these aren't exactly the same. torch.bmm must be numerically slightly
    # different from torch.baddbmm.
    assert torch.allclose(out[-(n_preds - 1):], last_xs, atol=5e-5)
    not_close = ~torch.isclose(out[-(n_preds - 1):], last_xs)
    print(f'Not "close": {100 * not_close.sum() / last_xs.numel():.4f}%')
    abs_diffs = (out[-(n_preds - 1):][not_close] - last_xs[not_close]).abs()
    print(f'Mean non-"close" absolute difference: {abs_diffs.mean()}')
    print(f'Max non-"close" absolute difference: {abs_diffs.max()}')


if __name__ == "__main__":
    main()
