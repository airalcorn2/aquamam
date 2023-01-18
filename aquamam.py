import numpy as np
import torch
import torch.nn.functional as F

from cached_layers import TransformerEncoder, TransformerEncoderLayer
from torch import nn


def get_labels(qs, bins):
    q_labels = torch.bucketize(qs, bins)
    return q_labels


def constrain_qys(qxs, bins, bin_bottoms, x):
    # Set probabilities of impossible bins to zero.
    qx_bins = torch.bucketize(qxs, bins)
    qx_bottoms = bin_bottoms[qx_bins]
    max_qys = torch.sqrt(1 - qx_bottoms**2)
    min_labels = torch.bucketize(-max_qys, bins)
    max_labels = torch.bucketize(max_qys, bins)
    idxs = torch.arange(len(bins)).unsqueeze(0).repeat(len(x), 1).to(x)
    idxs_mask = (idxs < min_labels) | (idxs > max_labels)
    x[idxs_mask] = -10000


def constrain_qzs(qxs, qys, bins, bin_bottoms, x):
    # Set probabilities of impossible bins to zero.
    qx_bins = torch.bucketize(qxs, bins)
    qx_bottoms = bin_bottoms[qx_bins]
    qy_bins = torch.bucketize(qys, bins)
    qy_bottoms = bin_bottoms[qy_bins]
    max_qzs = torch.sqrt(1 - qx_bottoms**2 - qy_bottoms**2)
    min_labels = torch.bucketize(-max_qzs, bins)
    max_labels = torch.bucketize(max_qzs, bins)

    idxs = torch.arange(len(bins)).unsqueeze(0).repeat(len(x), 1).to(x)
    idxs_mask = (idxs < min_labels) | (idxs > max_labels)
    x[idxs_mask] = -10000


def constrain_qs(qs, bins, bin_bottoms, x):
    constrain_qys(qs[:, 0:1], bins, bin_bottoms, x[:, 1])
    constrain_qzs(qs[:, 0:1], qs[:, 1:2], bins, bin_bottoms, x[:, 2])


def get_sample_tokens(which_val, qs, bins, bin_bottoms, x):
    if which_val == 1:
        constrain_qys(qs[:, 0:1], bins, bin_bottoms, x)

    elif which_val == 2:
        constrain_qzs(qs[:, 0:1], qs[:, 1:2], bins, bin_bottoms, x)

    probs = torch.softmax(x, dim=-1)
    tokens = torch.multinomial(probs, 1).flatten()
    if which_val < 2:
        qc_signs = torch.sign(bins[tokens])
        qc_bottoms = bin_bottoms[tokens]
        bin_width = bins[1] - bins[0]
        us = torch.rand(len(tokens)).to(qs)
        qs[:, which_val] = qc_signs * (qc_bottoms + us * bin_width)

    return tokens


def sample_qs(bin_bottoms, tokens, bins, bin_width):
    qs = []
    for which_val in range(tokens.shape[1]):
        bottoms = bin_bottoms[tokens[:, which_val]]
        signs = torch.sign(bins[tokens[:, which_val]])
        us = torch.rand(len(bottoms)).to(bottoms)
        qs.append(signs * (bottoms + us * bin_width))

    return torch.stack(qs).T


def convert_q_tokens_to_vals(tokens, bins, bin_bottoms):
    bin_width = bins[1] - bins[0]
    qs = sample_qs(bin_bottoms, tokens, bins, bin_width)
    norms = torch.norm(qs, dim=1)
    while norms.max() > 1:
        bad_norms = norms > 1
        qs[norms > 1] = sample_qs(bin_bottoms, tokens[bad_norms], bins, bin_width)
        norms = torch.norm(qs, dim=1)

    qws = torch.sqrt(1 - torch.sum(qs**2, dim=1))
    return torch.cat([qs, qws[:, None]], dim=1)


def run_beam_search(which_val, x, k, beam, bins, bin_bottoms, beam_log_ps):
    n_bins = len(bins)
    batch_size = len(x) // k
    if which_val == 0:
        log_ps = torch.log_softmax(x, dim=1)[::k]
        (beam_log_ps, idxs) = log_ps.topk(k, dim=1)
        beam[:, 0] = 2 * ((idxs.flatten() + 0.5) / n_bins - 0.5)

    else:
        if which_val == 1:
            constrain_qys(beam[:, 0:1], bins, bin_bottoms, x)

        elif which_val == 2:
            constrain_qzs(beam[:, 0:1], beam[:, 1:2], bins, bin_bottoms, x)

        log_ps = torch.log_softmax(x, dim=1)
        candidate_beam_log_ps = (beam_log_ps + log_ps).reshape(batch_size, -1)
        (beam_log_ps, idxs) = candidate_beam_log_ps.topk(k, dim=1)

        (rows, cols) = np.unravel_index(idxs.cpu().numpy(), log_ps.shape)
        rows = rows + k * np.arange(len(rows))[:, None]
        beam[:, :which_val] = beam[rows.flatten(), :which_val]

        col_vals = (torch.Tensor(cols).flatten().to(x) + 0.5) / n_bins
        beam[:, which_val] = 2 * (col_vals - 0.5)

    return beam_log_ps.flatten().unsqueeze(-1)


def finish_beam_search(beam, beam_log_ps, k):
    n_preds = beam.shape[1] - 1
    qws = torch.sqrt(1 - torch.sum(beam[:, :n_preds] ** 2, dim=1))
    qws[torch.isnan(qws)] = 0
    beam[:, n_preds] = qws
    beam = beam / torch.norm(beam, dim=1, keepdim=True)
    beam_log_ps = beam_log_ps.reshape(-1, k)
    beam = beam.reshape(beam_log_ps.shape[0], k, -1)
    return (beam, beam_log_ps)


def get_exact_densities(vals, bins, bin_bottoms, max_vals):
    val_bins = torch.bucketize(vals, bins)
    val_bottoms = bin_bottoms[val_bins]
    bin_width = bins[1] - bins[0]
    val_tops = val_bottoms + bin_width
    max_vals = torch.stack([max_vals, val_tops]).T
    max_vals = torch.min(max_vals, dim=1)[0]
    return 1 / (max_vals - val_bottoms)


def get_exact_qy_densities(qs, bins, bin_bottoms):
    max_qys = torch.sqrt(1 - qs[:, 0] ** 2)
    return get_exact_densities(qs[:, 1], bins, bin_bottoms, max_qys)


def get_exact_qz_densities(qs, bins, bin_bottoms):
    max_qzs = torch.sqrt(1 - qs[:, 0] ** 2 - qs[:, 1] ** 2)
    return get_exact_densities(qs[:, 2], bins, bin_bottoms, max_qzs)


class AQuaMaM(nn.Module):
    def __init__(self, toy_args, L, d_model, nhead, dropout, num_layers, n_bins):
        super().__init__()
        self.is_toy = toy_args["is_toy"]
        self.L = L
        mlps = []
        self.n_preds = n_preds = 3
        for i in range(n_preds - 1):
            mlp = nn.Sequential(
                nn.Linear(1 + 2 * self.L, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
            )
            mlps.append(mlp)

        self.mlps = nn.ModuleList(mlps)

        (img_size, patch_size) = (224, 16)
        self.n_patches = n_patches = (img_size // patch_size) ** 2
        if self.is_toy:
            n_cats = toy_args["max_pow"] + 1
            self.patch_embed = nn.Embedding(n_cats, n_patches * d_model)

        else:
            in_channels = 3
            self.patch_embed = nn.Conv2d(
                in_channels, d_model, kernel_size=patch_size, stride=patch_size
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        seq_len = n_patches + n_preds
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        dim_feedforward = 4 * d_model
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, F.gelu, 1e-6, norm_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        mask[:, : -(n_preds - 1)] = 0
        self.register_buffer("mask", mask)

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_bins))

        bins = torch.linspace(-1, 1, n_bins + 1)[1:]
        self.register_buffer("bins", bins)
        bin_width = bins[1] - bins[0]
        bin_bottoms = torch.cat([bins[bins <= 0].abs(), bins[bins > 0] - bin_width])
        self.register_buffer("bin_bottoms", bin_bottoms)

    def get_img_embeds(self, imgs):
        if self.is_toy:
            return self.patch_embed(imgs).reshape(len(imgs), self.n_patches, -1)
        else:
            return self.patch_embed(imgs).flatten(2).transpose(1, 2)

    def get_encoded_qcs(self, qs, i):
        qcs = qs[:, i : i + 1]
        qcs_encoded = [qcs]
        for l_pos in range(self.L):
            qcs_encoded.append(torch.sin(2**l_pos * torch.pi * qcs))
            qcs_encoded.append(torch.cos(2**l_pos * torch.pi * qcs))

        return self.mlps[i](torch.cat(qcs_encoded, dim=-1)).unsqueeze(1)

    def get_beam(self, imgs, k):
        x = self.get_img_embeds(imgs).repeat_interleave(k, dim=0)
        x = torch.cat([x, self.cls_token.expand(len(x), -1, -1)], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]

        mem_kvs = []
        x = x.permute(1, 0, 2)
        for mod in self.transformer.layers:
            (x, mem_kv) = mod(x, return_kv=True)
            mem_kvs.append(mem_kv)

        x = self.head(x.permute(1, 0, 2)[:, -1])

        beam = torch.zeros(len(imgs) * k, self.n_preds + 1).to(x)
        beam_log_ps = run_beam_search(0, x, k, beam, self.bins, self.bin_bottoms, None)

        for which_val in range(1, self.n_preds):
            x = self.get_encoded_qcs(beam, which_val - 1)
            x = x + self.pos_embed[:, -(self.n_preds - which_val)].unsqueeze(1)
            x = x.permute(1, 0, 2)

            for (mod_idx, mod) in enumerate(self.transformer.layers):
                (x, mem_kv) = mod(x, mem_kv=mem_kvs[mod_idx], return_kv=True)
                mem_kvs[mod_idx] = mem_kv

            x = self.head(x.permute(1, 0, 2).squeeze(1))
            beam_log_ps = run_beam_search(
                which_val, x, k, beam, self.bins, self.bin_bottoms, beam_log_ps
            )

        return finish_beam_search(beam, beam_log_ps, k)

    def beam_search(self, imgs, k):
        (beam, beam_log_ps) = self.get_beam(imgs, k)
        beam_maxes = beam_log_ps.argmax(dim=1)
        return beam[torch.arange(len(imgs)), beam_maxes]

    def sample(self, imgs):
        x = self.get_img_embeds(imgs)
        x = torch.cat([x, self.cls_token.expand(len(x), -1, -1)], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]

        mem_kvs = []
        x = x.permute(1, 0, 2)
        for mod in self.transformer.layers:
            (x, mem_kv) = mod(x, return_kv=True)
            mem_kvs.append(mem_kv)

        x = self.head(x.permute(1, 0, 2)[:, -1])

        qs = torch.zeros(len(imgs), self.n_preds).to(x)
        sample_tokens = [get_sample_tokens(0, qs, self.bins, self.bin_bottoms, x)]

        for which_val in range(1, self.n_preds):
            x = self.get_encoded_qcs(qs, which_val - 1)
            x = x + self.pos_embed[:, -(self.n_preds - which_val)].unsqueeze(1)
            x = x.permute(1, 0, 2)

            for (mod_idx, mod) in enumerate(self.transformer.layers):
                (x, mem_kv) = mod(x, mem_kv=mem_kvs[mod_idx], return_kv=True)
                mem_kvs[mod_idx] = mem_kv

            x = self.head(x.permute(1, 0, 2).squeeze(1))
            sample_tokens.append(
                get_sample_tokens(which_val, qs, self.bins, self.bin_bottoms, x)
            )

        sample_tokens = torch.stack(sample_tokens).T
        sample_vals = convert_q_tokens_to_vals(
            sample_tokens, self.bins, self.bin_bottoms
        )
        return (sample_tokens, sample_vals)

    def forward(self, imgs, qs):
        x = self.get_img_embeds(imgs)
        xs = [x, self.cls_token.expand(len(x), -1, -1)]
        for i in range(self.n_preds - 1):
            xs.append(self.get_encoded_qcs(qs, i))

        x = torch.cat(xs, dim=1)
        x = x + self.pos_embed
        x = self.transformer(x.permute(1, 0, 2), self.mask).permute(1, 0, 2)

        x = self.head(x[:, -self.n_preds :])
        constrain_qs(qs, self.bins, self.bin_bottoms, x)
        return x
