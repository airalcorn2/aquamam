import numpy as np
import torch
import torch.nn.functional as F

from cached_layers import TransformerEncoder, TransformerEncoderLayer
from torch import nn

LNSQRT2PI = np.log(np.sqrt(2 * np.pi))


def get_sample_qcs(qs, x, which_qc):
    n_comps = x.shape[-1] // 3
    probs = torch.softmax(x[:, :n_comps], dim=1)
    comps = torch.multinomial(probs, 1).flatten()

    idxs = torch.arange(len(comps))
    mus = x[:, n_comps : 2 * n_comps][idxs, comps]
    sds = torch.exp(x[:, 2 * n_comps :][idxs, comps])
    ss = torch.normal(mus, sds)

    ubs = torch.sqrt(1 - torch.sum(qs[:, :which_qc] ** 2, dim=1))
    qcs = -ubs + 2 * ubs * torch.sigmoid(ss)
    return qcs


def get_pre_lls(x, qs):
    n_comps = x.shape[-1] // 3
    lls = 0
    for i in range(3):
        ubs = torch.sqrt(1 - torch.sum(qs[:, :i] ** 2, dim=1))

        qcs = qs[:, i]
        ss = torch.log(qcs + ubs) - torch.log(ubs - qcs)
        ss = ss.unsqueeze(1).repeat(1, n_comps)

        log_pis = torch.log_softmax(x[:, i, :n_comps], dim=1)

        mus = x[:, i, n_comps : 2 * n_comps]
        log_stds = x[:, i, 2 * n_comps :]
        log_ps = -log_stds - LNSQRT2PI - 0.5 * ((ss - mus) / torch.exp(log_stds)) ** 2

        lls += torch.logsumexp(log_pis + log_ps, dim=1)

    return lls


def get_full_lls(x, qs):
    lls = get_pre_lls(x, qs) + torch.log(qs[:, 3])
    for i in range(3):
        ubs = torch.sqrt(1 - torch.sum(qs[:, :i] ** 2, dim=1))
        qcs = qs[:, i]
        lls += torch.log(torch.abs(2 * ubs / (ubs**2 - qcs**2)))

    return lls


class AQuaMaMMoG(nn.Module):
    def __init__(self, toy_args, L, d_model, nhead, dropout, num_layers, n_comps):
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

        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 3 * n_comps)
        )

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

        qs = torch.zeros(len(imgs), self.n_preds + 1).to(x)
        qs[:, 0] = get_sample_qcs(qs, x, 0)
        for which_qc in range(1, self.n_preds):
            x = self.get_encoded_qcs(qs, which_qc - 1)
            x = x + self.pos_embed[:, -(self.n_preds - which_qc)].unsqueeze(1)
            x = x.permute(1, 0, 2)

            for (mod_idx, mod) in enumerate(self.transformer.layers):
                (x, mem_kv) = mod(x, mem_kv=mem_kvs[mod_idx], return_kv=True)
                mem_kvs[mod_idx] = mem_kv

            x = self.head(x.permute(1, 0, 2).squeeze(1))
            qs[:, which_qc] = get_sample_qcs(qs, x, which_qc)

        qs[:, self.n_preds] = torch.sqrt(1 - torch.sum(qs[:, : self.n_preds] ** 2, 1))
        qs[torch.isnan(qs)] = 0
        qs = qs / torch.norm(qs, dim=1, keepdim=True)
        return qs

    def forward(self, imgs, qs):
        x = self.get_img_embeds(imgs)
        xs = [x, self.cls_token.expand(len(x), -1, -1)]
        for i in range(self.n_preds - 1):
            xs.append(self.get_encoded_qcs(qs, i))

        x = torch.cat(xs, dim=1)
        x = x + self.pos_embed
        x = self.transformer(x.permute(1, 0, 2), self.mask).permute(1, 0, 2)

        x = self.head(x[:, -self.n_preds :])
        return x
