import logging
import os
import sys

sys.path.append("../")

import torch
import dgl
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.nn import Module
from tqdm import tqdm
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset
# from gated_delta_net import GatedDeltaNet
# from mamba_ssm import  Mamba2
from utils.DepMamba import DepMamba2, CNNEncoderLayer
import time
from utils.ka_gnn import KA_GNN, KA_GNN_two
from torch.distributions import Categorical
import random
import numpy as np
from utils.PMMA import MultiHeadLinearAttention
from utils.PGCA import GuidedCrossAttention
from utils.base import *
from utils.vmf_loss import vMFLoss


LOGGER = logging.getLogger(__name__)

def dsu(x, mask=None, eps=1e-6, p=0.5):
    """Distribution Shift Uncertainty — 在序列维度上扰动特征统计量，模拟 domain shift。
    mask: (B, L) bool, True=有效token, False=padding, 用于排除填充位
    """
    if not torch.is_grad_enabled() or random.random() > p:
        return x

    def _reparameterize(mu, std):
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def _sqrtvar(x):
        t = (x.var(dim=0, keepdim=True) + eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    if mask is not None:
        m = mask.unsqueeze(-1).to(x.dtype)
        valid = m.sum(dim=1).clamp(min=1)
        mean = (x * m).sum(dim=1) / valid
        var = ((x - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / valid
    else:
        mean = x.mean(dim=1, keepdim=False)
        var = x.var(dim=1, keepdim=False, unbiased=False)

    std = (var + eps).sqrt()

    sqrtvar_mu = _sqrtvar(mean)
    sqrtvar_std = _sqrtvar(std)

    beta = _reparameterize(mean, sqrtvar_mu)
    gamma = _reparameterize(std, sqrtvar_std)

    if mask is not None:
        x_pad = x * (1 - m)  # 保存 padding 原始值
    x = (x - mean.unsqueeze(1)) / std.unsqueeze(1)
    x = x * gamma.unsqueeze(1) + beta.unsqueeze(1)
    if mask is not None:
        x = x * m + x_pad  # 有效 token 用扰动值，padding 保持原值
    return x

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def message_func(edges):
    return {'feat': edges.data['feat']}


def reduce_func(nodes):
    num_edges = nodes.mailbox['feat'].size(1)
    agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges
    return {'agg_feats': agg_feats}


def update_node_features(g):
    g.send_and_recv(g.edges(), message_func, reduce_func)

    g.ndata['feat'] = torch.cat((g.ndata['feat'], g.ndata['agg_feats']), dim=1)

    return g


class PMLDTI(nn.Module):
    def __init__(self, prot_out_dim, disease_out_dim, args):
        super(PMLDTI, self).__init__()
        self.hidden_dim = 128
        self.fusion = args.fusion
        self.norm = 'ln' if getattr(args, 'task', None) == 'cluster' else 'bn'
        self.use_dsu = getattr(args, 'task', None) == 'cluster'
        self.use_mask = getattr(args, 'use_mask', True)
        self.depmamba = DepMamba2(**args.mmmamba, norm=self.norm)


        if self.fusion == "CAN":
            self.can_layer_1 = CAN_Layer(hidden_dim=512, num_heads=8, args=args)
            self.can_layer_2 = CAN_Layer(hidden_dim=256, num_heads=8, args=args)
            self.mlp_classifier = MlPdecoder_CAN(input_dim=512)
        elif self.fusion == "BAN":
            self.ban_layer_1 = weight_norm(BANLayer(256, 256, 256, 2), name='h_mat', dim=None)
            self.ban_layer_2 = weight_norm(BANLayer(256, 256, 256, 2), name='h_mat', dim=None)
            self.mlp_classifier = MlPdecoder_CAN(input_dim=512)
        elif self.fusion == "Nan":
            self.mlp_classifier_nan = MlPdecoder_CAN(input_dim=512)

        self.ka_gnn = KA_GNN(in_feat=54, hidden_feat=self.hidden_dim, out_feat=self.hidden_dim, out=1,
                             grid_feat=2, num_layers=3, pooling='avg', use_bias=True)

        num_heads = getattr(args, 'num_heads', 1)
        self.sequence_fusion = DrugFusion_Layer(hidden_dim=self.hidden_dim, num_heads=num_heads, args=args)
        self.graph_fusion = DrugFusion_Layer(hidden_dim=self.hidden_dim, num_heads=num_heads, args=args)



        #GatedCNN
        self.gated_cnn_sap = GatedCNN(
            protein_dim=1280,
            hid_dim=self.hidden_dim,
            n_layers=3,
            kernel_size=1,
            dropout=0.1
        )

        self.gated_cnn_esm = GatedCNN(
            protein_dim=960,
            hid_dim=self.hidden_dim,
            n_layers=3,
            kernel_size=1,
            dropout=0.1
        )

        self.gated_cnn_drug = GatedCNN(
            protein_dim=768,
            hid_dim=self.hidden_dim,
            n_layers=3,
            kernel_size=1,
            dropout=0.1
        )

        ###loss_cls, loss_ent
        self.input_dim = self.hidden_dim * 2
        self.beta = args.beta * self.input_dim
        self.mid_dim = self.input_dim
        self.num_views = 2
        self.num_classes = 2
        self.txt_logit = LogitCollector(self.input_dim, self.input_dim, self.beta, num_layer1=1, num_layer2=2, norm=self.norm)
        self.img_logit = LogitCollector(self.input_dim, self.input_dim, self.beta, num_layer1=1, num_layer2=2, norm=self.norm)

        self.classifier_vmf = vMFLoss(self.mid_dim * self.num_views, self.num_classes)


    def dgl_split(self, bg, feats):
        max_num_nodes = int(bg.batch_num_nodes().max())
        batch = torch.cat(
            [torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],
            dim=1).reshape(-1).type(torch.long).to(bg.device)
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def forward(self, prot_embed, prot_esmc, drug_embed, drug_graph, prot_mask, drug_mask, choice, ret=False):
        start_time = time.time()
        drug_graph.ndata['feat'] = drug_graph.ndata['feat'].to(torch.float32)
        drug_graph.edata['feat'] = drug_graph.edata['feat'].to(torch.float32)

        prot_embed = prot_embed.to(torch.float32)
        prot_esmc = prot_esmc.to(torch.float32)
        drug_embed = drug_embed.to(torch.float32)

        if self.use_dsu and self.training:
            prot_embed = dsu(prot_embed)
            prot_esmc = dsu(prot_esmc)
            drug_embed = dsu(drug_embed, drug_mask)

        # 记录开始时间
        # start_time = time.time()
        if self.fusion == "Nan":
            prot_embed = prot_embed.mean(1)  # query : [batch_size, hidden]
            drug_embed = drug_embed.mean(1)  # query : [batch_size, hidden]
            joint_embed = torch.cat([prot_embed, drug_embed], dim=1)
            score = self.mlp_classifier_nan(joint_embed)
        else:
            prot_embed_esm = self.gated_cnn_esm(prot_esmc)
            prot_embed_sap = self.gated_cnn_sap(prot_embed)

            g = update_node_features(drug_graph)
            x = g.ndata['feat']
            out = self.ka_gnn(g, x)
            drug_graph_out = self.dgl_split(g, out)

            drug_embed = self.gated_cnn_drug(drug_embed)
            num_nodes = g.batch_num_nodes()
            drug_mask_graph = torch.arange(drug_graph_out.shape[1], device=drug_graph_out.device).unsqueeze(0) < num_nodes.unsqueeze(1)
            if self.use_dsu and self.training:
                drug_graph_out = dsu(drug_graph_out, drug_mask_graph)

            if self.use_mask:
                graph_embed = self.graph_fusion(prot_embed_sap, drug_graph_out, prot_mask, drug_mask_graph)
                seq_embed = self.sequence_fusion(prot_embed_esm, drug_embed, prot_mask, drug_mask)
            else:
                graph_embed = self.graph_fusion(prot_embed_sap, drug_graph_out)
                seq_embed = self.sequence_fusion(prot_embed_esm, drug_embed)

            graph_embed, seq_embed = self.depmamba(graph_embed, seq_embed)

            txt_f = seq_embed
            img_f = graph_embed

            means = {}
            kappas = {}
            weights = {}

            cb_list = []

            means[0], kappas[0] = self.img_logit(img_f)
            weights[0] = kappas[0]
            cb_list.append(weights[0])

            means[1], kappas[1] = self.txt_logit(txt_f)
            weights[1] = kappas[1]
            cb_list.append(weights[1])

            w_all = (torch.concat(cb_list, dim=-1))
            w_all = w_all / w_all.sum(-1, keepdim=True)

            logits = 0
            preds, means_w = {}, []
            weight = self.classifier_vmf.classifier.weight
            prototypes = []
            for view in range(self.num_views):
                w = w_all[:, view: view + 1]
                means_w.append(w * means[view])
                prototype = F.normalize(weight[:, view * self.mid_dim: (view + 1) * self.mid_dim], 2, dim=-1)
                preds[view] = self.classifier_vmf(means[view], kappas[view], prototype, norm=False)
                weights[view] = w
                prototypes.append(prototype)

            kappa = sum(kappas.values()) / self.num_views  # sum([k.pow(2) for k in kappas.values()]).sqrt()
            feat = torch.cat(means_w, dim=-1)
            feat = feat / feat.norm(dim=-1, keepdim=True).detach()
            prototype = torch.cat(prototypes, dim=-1) / np.sqrt(self.num_views)
            logits = self.classifier_vmf(feat, kappa, prototype, norm=False)


            # pdf_test
            if choice == 'pdf_test':
                if not self.training:
                    cross_entropy, means_w, self_entropy, pred_all = [], [], [], logits.softmax(-1)
                    for view in range(self.num_views):
                        pred = preds[view].softmax(-1)
                        cross_entropy.append((pred * pred_all.clamp_min(1e-7).log()).sum(-1, keepdim=True))
                        self_entropy.append((pred * pred.clamp_min(1e-7).log()).sum(-1, keepdim=True))
                    cross_entropy = torch.concat(cross_entropy, dim=-1).softmax(-1)
                    self_entropy = torch.concat(self_entropy, dim=-1).softmax(-1)
                    rel = (cross_entropy + w_all) / 2.

                    for view in range(self.num_views):
                        w = rel[:, view: view + 1]
                        means_w.append(w * means[view])

                    feat = F.normalize(torch.cat(means_w, dim=-1), 2, dim=-1)
                    logits = self.classifier_vmf(feat, kappa, prototype, norm=False)
                    if ret:
                        return logits, preds, weights, kappas, self_entropy, cross_entropy, rel
            end_time = time.time()

            return logits, preds, weights


class DrugFusion_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, args):
        super(DrugFusion_Layer, self).__init__()
        self.v_gca = GuidedCrossAttention(embed_dim=hidden_dim, num_heads=num_heads)
        # self.v_gca = CAN_Layer(hidden_dim=hidden_dim, num_heads=1, args=args)
        self.v_mhla = MultiHeadLinearAttention(d_model=hidden_dim * 2, d_diff=hidden_dim * 8, nhead=8,
                                               dropout=0.2, activation='gelu')

        self.v_gca_norm = nn.LayerNorm(hidden_dim * 2)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, protein, drug, mask_prot=None, mask_drug=None):
        # mask_drug: (N, S) bool, True=真实节点/token；key_padding_mask 语义为 True=忽略(padding)，故取反
        key_padding_mask = (~mask_drug) if mask_drug is not None else None
        prot_embedding, weight = self.v_gca(
            protein.permute(1, 0, 2), drug.permute(1, 0, 2), drug.permute(1, 0, 2),
            key_padding_mask=key_padding_mask,
        )
        prot_embedding = prot_embedding.permute(1, 0, 2)
        weight = weight.sum(dim=(0, 1))    #注意力权重
        mv = torch.cat((protein, prot_embedding), 2)

        return mv
        # mv = self.linear(mv)
        hv = mv
        mv = self.v_mhla(mv)
        mv = mv + hv
        mv = self.v_gca_norm(mv)
        # mv = self.linear(mv)
        return mv


class CAN_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, args):
        super(CAN_Layer, self).__init__()
        self.agg_mode = args.agg_mode
        self.group_size = args.group_size  # Control Fusion Scale
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

    def forward(self, protein, drug, mask_prot, mask_drug):
        # Group embeddings before applying multi-head attention
        protein_grouped, mask_prot_grouped = self.group_embeddings(protein, mask_prot, self.group_size)
        drug_grouped, mask_drug_grouped = self.group_embeddings(drug, mask_drug, self.group_size)

        # print("protein_grouped:", protein_grouped.shape)
        # print("mask_prot_grouped:", mask_prot_grouped.shape)

        # Compute queries, keys, values for both protein and drug after grouping
        query_prot = self.apply_heads(self.query_p(protein_grouped), self.num_heads, self.head_size)

        key_drug = self.apply_heads(self.key_d(drug_grouped), self.num_heads, self.head_size)
        value_drug = self.apply_heads(self.value_d(drug_grouped), self.num_heads, self.head_size)

        # Compute attention scores
        logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_drug)
        # print("logits_pp:", logits_pp.shape)

        alpha_pd = self.alpha_logits(logits_pd, mask_prot_grouped, mask_drug_grouped)

        # prot_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_pp, value_prot).flatten(-2) +
        #            torch.einsum('blkh, bkhd->blhd', alpha_pd, value_drug).flatten(-2)) / 2
        # drug_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
        #            torch.einsum('blkh, bkhd->blhd', alpha_dd, value_drug).flatten(-2)) / 2

        prot_embedding = torch.einsum('blkh, bkhd->blhd', alpha_pd, value_drug).flatten(-2)
        return prot_embedding


        # Continue as usual with the aggregation mode
        if self.agg_mode == "cls":
            prot_embed = prot_embedding[:, 0]  # query : [batch_size, hidden]
            drug_embed = drug_embedding[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            prot_embed = prot_embedding.mean(1)  # query : [batch_size, hidden]
            drug_embed = drug_embedding.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            prot_embed = (prot_embedding * mask_prot_grouped.unsqueeze(-1)).sum(1) / mask_prot_grouped.sum(
                -1).unsqueeze(-1)
            drug_embed = (drug_embedding * mask_drug_grouped.unsqueeze(-1)).sum(1) / mask_drug_grouped.sum(
                -1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        # print("prot_embed:", prot_embed.shape)

        query_embed = torch.cat([prot_embed, drug_embed], dim=1)

        # print("query_embed:", query_embed.shape)
        return query_embed



class MlPdecoder_CAN(nn.Module):
    def __init__(self, input_dim):
        super(MlPdecoder_CAN, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.bn2 = nn.BatchNorm1d(input_dim // 2)
        self.fc3 = nn.Linear(input_dim // 2, input_dim // 4)
        self.bn3 = nn.BatchNorm1d(input_dim // 4)
        self.output = nn.Linear(input_dim // 4, 2)

        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        # x = self.bn1(torch.relu(self.fc1(x)))
        # x = self.bn2(torch.relu(self.fc2(x)))
        # x = self.bn3(torch.relu(self.fc3(x)))
        # x = torch.sigmoid(self.output(x))
        # x = self.output(x)
        x = self.fc(x)
        return x


class BANLayer(nn.Module):
    """ Bilinear attention network
    Modified from https://github.com/peizhenbai/DrugBAN/blob/main/ban.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        # print("v_num", v_num)
        # print("v_num ", v_num)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            # print("v_", v_.shape)
            # print("q_ ", q_.shape)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            # print("Attention map_1",att_maps.shape)
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
            # print("Attention map_2",att_maps.shape)
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
            # print("Attention map_softmax", att_maps.shape)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

import gc

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BatchFileDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # start_time = time.time()
        batch_file = self.file_list[idx]
        data = torch.load(batch_file)
        gc.collect()
        # end_time = time.time()

        # elapsed_time = end_time - start_time
        # print(f"Forward pass completed in {elapsed_time:.4f} seconds")
        return data['prot'], data['prot_esmc'], data['drug'], data['drug_graph'], data['prot_mask'], data['drug_mask'], \
        data['y']


class GatedCNN(nn.Module):
    """Convolutional encoder for protein feature extraction."""

    def __init__(self, protein_dim: int, hid_dim: int, n_layers: int,
                 kernel_size: int, dropout: float):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([
            nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(protein_dim, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of encoder.

        Args:
            protein: Protein embeddings [batch_size, seq_len, protein_dim]

        Returns:
            Encoded features [batch_size, seq_len, hid_dim]
        """
        # Linear projection
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)  # [batch_size, hid_dim, seq_len]

        # Convolutional layers with GLU activation
        for conv in self.convs:
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)  # Gated Linear Unit
            conved = (conved + conv_input) * self.scale  # Residual connection
            conv_input = conved

        # Back to original shape and normalize
        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)

        return conved