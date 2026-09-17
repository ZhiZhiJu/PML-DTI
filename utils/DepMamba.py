"""DepMamba implementation.
Authors
-------
* Jiaxin Ye 2024
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
    RelPosMHAXL,
)
from speechbrain.nnet.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

# Mamba
from mamba_ssm import Mamba
from .mamba.bimamba import Mamba as BiMamba
from .mamba.mm_bimamba import Mamba as MMBiMamba
from .base import BaseNet
from .mamba.MCOMamba import CrossMamba
from utils.PMMA import MultiHeadLinearAttention


class MMMambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        self.a_downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(d_model),
        )

    def forward(
        self,
        a_x, v_x,
        a_inference_params = None,
        v_inference_params = None
    ):

        a_out1, v_out1 = self.mamba(a_x, v_x,a_inference_params,v_inference_params)
        a_out = a_x + self.norm1(a_out1)
        v_out = v_x + self.norm2(v_out1)

        return a_out, v_out

class MMCNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
        norm='bn',
    ):
        super().__init__()

        def make_norm(ch):
            return nn.GroupNorm(1, ch) if norm == 'ln' else nn.BatchNorm1d(ch)

        self.a_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.a_bn = make_norm(output_size)

        self.v_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.v_bn = make_norm(output_size)

        self.relu = nn.ReLU()

        self.a_drop = nn.Dropout(dropout)
        self.v_drop = nn.Dropout(dropout)

        self.a_net = nn.Sequential(self.a_conv, self.a_bn, self.relu, self.a_drop)
        self.v_net = nn.Sequential(self.v_conv, self.v_bn, self.relu, self.v_drop)

        if input_size != output_size:
            self.a_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
            self.v_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.a_skipconv = None
            self.v_skipconv = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_conv.weight.data)
        nn.init.xavier_uniform_(self.v_conv.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, xa, xv):
        a_out = self.a_net(xa)
        v_out = self.v_net(xv)
        if self.a_skipconv is not None:
            xa = self.a_skipconv(xa)
        if self.v_skipconv is not None:
            xv = self.v_skipconv(xv)
        a_out = a_out+xa
        v_out = v_out+xv
        return a_out, v_out

class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)


    def forward(
        self,
        x, inference_params = None
    ):
        out = x + self.norm1(self.mamba(x, inference_params))
        return out

class CNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv1d(output_size, output_size, 5, padding=2, dilation=dilation, bias=False)
        # self.bn2 = nn.BatchNorm1d(output_size)
        # self.relu2 = nn.ReLU()

        self.drop = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)

        if input_size != output_size:
            self.conv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.conv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, x):
        out = self.net(x)
        if self.conv is not None:
            x = self.conv(x)
        out = out+x
        return out

class CoSSM(nn.Module):
    """This class implements the CoSSM encoder.
    """
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256,512,512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        kernel_size = 3,
        causal=False,
        mamba_config=None,
        norm='bn'
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        a_norm_list = []
        v_norm_list = []
        att_list = []
        # print(output_sizes)
        for i in range(len(output_sizes)):
            cnn_list.append(MMCNNEncoderLayer(
                    input_size = input_size if i<1 else output_sizes[i-1],
                    output_size = output_sizes[i],
                    dropout=dropout,
                    norm=norm
                ))
            mamba_list.append(MMMambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config,
                ))
            a_norm_list.append(nn.LayerNorm(output_sizes[i]))
            v_norm_list.append(nn.LayerNorm(output_sizes[i]))

            att_list.append(Attention(output_sizes[i], 2, 1))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)
        self.a_norm_layers = torch.nn.ModuleList(a_norm_list)
        self.v_norm_layers = torch.nn.ModuleList(v_norm_list)
        self.att_layers = torch.nn.ModuleList(att_list)


    def forward(
        self,
        a_x, v_x,
        a_inference_params = None,
        v_inference_params = None
    ):
        a_out = a_x
        v_out = v_x

        for cnn_layer, a_norm, v_norm, mamba_layer, att_layer in zip(
                self.cnn_layers, self.a_norm_layers, self.v_norm_layers, self.mamba_layers, self.att_layers
        ):
            a_out, v_out  = cnn_layer(a_out.permute(0,2,1), v_out.permute(0,2,1))
            a_out = a_out.permute(0,2,1)
            v_out = v_out.permute(0,2,1)

            # a_out, v_out = att_layer(a_out, v_out)
            a_out = a_norm(a_out)
            v_out = v_norm(v_out)

            a_out, v_out = mamba_layer(
                a_out, v_out,
                a_inference_params = a_inference_params,
                v_inference_params = v_inference_params
            )

            # a_out = a_norm(a_out)
            # v_out = v_norm(v_out)

        return a_out, v_out

class EnSSM(nn.Module):
    """This class implements the EnSSM encoder.
    """
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256,512,512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        # print(output_sizes)
        for i in range(len(output_sizes)):
            cnn_list.append(CNNEncoderLayer(
                    input_size = input_size if i<1 else output_sizes[i-1],
                    output_size = output_sizes[i],
                    dropout=dropout
                ))
            mamba_list.append(MambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config,
                ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)


    def forward(
        self,
        x,
        inference_params = None,
    ):
        out = x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out  = cnn_layer(out.permute(0,2,1))
            out = out.permute(0,2,1)
            out = mamba_layer(
                out,
                inference_params = inference_params,
            )

        return out

class DepMamba2(BaseNet):

    def __init__(self, audio_input_size=161, video_input_size=161, mm_input_size=128, mm_output_sizes=[256,64], d_ffn=1024, num_layers=8, dropout=0.1, activation='Swish', causal=False, mamba_config=None, norm='bn'):
        super().__init__()

        self.cossm_encoder = CoSSM(num_layers,
                                         mm_input_size,
                                    mm_output_sizes,
                                    d_ffn,
                                    activation=activation,
                                    dropout=dropout,
                                    causal=causal,
                                    mamba_config=mamba_config,
                                    norm=norm)

        self.conv_audio_1 = nn.Conv1d(audio_input_size, mm_input_size * 2, 1, padding=0, dilation=1, bias=False)
        self.conv_video_1 = nn.Conv1d(video_input_size, mm_input_size * 2, 1, padding=0, dilation=1, bias=False)

        self.conv_audio_2 = nn.Conv1d(mm_input_size * 2, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.conv_video_2 = nn.Conv1d(mm_input_size * 2, mm_input_size, 1, padding=0, dilation=1, bias=False)

        self.enssm_encoder = EnSSM(num_layers,
                                    mm_output_sizes[-1] * 2,
                                    [mm_output_sizes[-1] * 2],
                                    d_ffn,
                                    activation=activation,
                                    dropout=dropout,
                                    causal=causal,
                                    mamba_config=mamba_config)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.output = nn.Linear(mm_output_sizes[-1], 1)
        self.m = nn.Sigmoid()
        self.v_gca_norm = nn.LayerNorm(mm_output_sizes[-1])
        # self.MCO = CrossMamba(mm_input_size)

        self.rescnnlayer = MMCNNEncoderLayer(
            input_size=mm_input_size,
            output_size=mm_input_size,
            dropout=dropout
        )

        nn.init.xavier_uniform_(self.conv_audio_1.weight.data)
        nn.init.xavier_uniform_(self.conv_video_1.weight.data)
        nn.init.xavier_uniform_(self.conv_audio_2.weight.data)
        nn.init.xavier_uniform_(self.conv_video_2.weight.data)


    def feature_extractor(self, xa, xv,  padding_mask=None, a_inference_params = None, v_inference_params = None):
        # xa = x[:, :, 136:]
        # xv = x[:, :, :136]
        # xa = self.conv_audio_1(xa.permute(0,2,1)).permute(0,2,1)
        # xv = self.conv_video_1(xv.permute(0,2,1)).permute(0,2,1)
        #
        # xa = self.conv_audio_2(xa.permute(0, 2, 1)).permute(0, 2, 1)
        # xv = self.conv_video_2(xv.permute(0, 2, 1)).permute(0, 2, 1)
        #
        #
        # xa_1, xv_1 =  self.rescnnlayer(xa.permute(0, 2, 1), xv.permute(0, 2, 1))
        # xa_1 = xa_1.permute(0,2,1)
        # xv_1 = xv_1.permute(0,2,1)

        # xa_1 = xa + xa_1
        # xv_1 = xv + xv_1


        xa, xv = self.cossm_encoder(xa, xv, a_inference_params, v_inference_params)

        #xa = xa + xa_1
        #xv = xv + xv_1

        # xa_2, xv_2 = self.MCO(xa_1, xv_1)
        # print(xa.shape)
        # print(xv.shape)

        # xa = xa_1 + xa + xa_2
        # xv = xv_1 + xv + xv_2
        # xa = xa + xa_2
        # xv = xv + xv_2




        # x = torch.cat([xa,xv],dim=-1)
        #x = xa + xv
        # x = self.enssm_encoder(x)
        # x = x * (padding_mask.unsqueeze(-1).float())
        # if padding_mask is not None:
        #      x = x * (padding_mask.unsqueeze(-1).float())
        #      x = x.sum(dim=1) / (padding_mask.unsqueeze(-1).float()
        #                          ).sum(dim=1, keepdim=False)  # Compute average
        # else:
        #      x = self.pool(x.permute(0,2,1)).squeeze(-1)

        #x = self.v_gca_norm(x)
        return xa, xv

    def forward(self, xa, xv, mask=None):
        xa, xv = self.feature_extractor(xa, xv, mask)
        return xa.mean(1), xv.mean(1)

    def classifier(self, x):
        return self.output(x)


class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads, group_size):
        super(Attention, self).__init__()
        self.group_size = group_size  # Control Fusion Scale
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def apply_heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def group_embeddings(self, x, mask, group_size):
        N, L, D = x.shape
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2)
        mask_grouped = mask.view(N, groups, group_size).any(dim=2)
        return x_grouped, mask_grouped

    def forward(self, protein, drug, mask_prot=None, mask_drug=None):
        # Group embeddings before applying multi-head attention
        mask_prot = torch.ones((protein.shape[0], protein.shape[1])).to(protein.device)
        mask_drug = torch.ones((drug.shape[0], drug.shape[1])).to(protein.device)
        protein_grouped, mask_prot_grouped = self.group_embeddings(protein, mask_prot, self.group_size)
        drug_grouped, mask_drug_grouped = self.group_embeddings(drug, mask_drug, self.group_size)

        # print("protein_grouped:", protein_grouped.shape)
        # print("mask_prot_grouped:", mask_prot_grouped.shape)

        # Compute queries, keys, values for both protein and drug after grouping
        query_prot = self.apply_heads(self.query_p(protein_grouped), self.num_heads, self.head_size)
        key_prot = self.apply_heads(self.key_p(protein_grouped), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein_grouped), self.num_heads, self.head_size)

        query_drug = self.apply_heads(self.query_d(drug_grouped), self.num_heads, self.head_size)
        key_drug = self.apply_heads(self.key_d(drug_grouped), self.num_heads, self.head_size)
        value_drug = self.apply_heads(self.value_d(drug_grouped), self.num_heads, self.head_size)

        # Compute attention scores
        logits_pp = torch.einsum('blhd, bkhd->blkh', query_prot, key_prot)
        #logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_drug)
        #logits_dp = torch.einsum('blhd, bkhd->blkh', query_drug, key_prot)
        logits_dd = torch.einsum('blhd, bkhd->blkh', query_drug, key_drug)
        # print("logits_pp:", logits_pp.shape)

        alpha_pp = self.alpha_logits(logits_pp, mask_prot_grouped, mask_prot_grouped)
        #alpha_pd = self.alpha_logits(logits_pd, mask_prot_grouped, mask_drug_grouped)
        #alpha_dp = self.alpha_logits(logits_dp, mask_drug_grouped, mask_prot_grouped)
        alpha_dd = self.alpha_logits(logits_dd, mask_drug_grouped, mask_drug_grouped)


        prot_embedding = torch.einsum('blkh, bkhd->blhd', alpha_pp, value_prot).flatten(-2)
        drug_embedding = torch.einsum('blkh, bkhd->blhd', alpha_dd, value_drug).flatten(-2)

        return prot_embedding, drug_embedding