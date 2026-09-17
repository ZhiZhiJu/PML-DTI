import abc

import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.PMMA import MultiHeadLinearAttention


class BaseNet(nn.Module, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def feature_extractor(self, x, mask=None):
        pass

    @abc.abstractmethod
    def classifier(self, x):
        pass

    def forward(self, xa, xv, mask=None):
        x = self.feature_extractor(xa, xv,  mask)
        # x = self.classifier(x)
        return x


class LogitCollector(nn.Module):
    def __init__(self, input_dim, mid_dim, beta, num_layer1=2, num_layer2=3, norm='bn'):
        super(LogitCollector, self).__init__()
        Norm = nn.LayerNorm if norm == 'ln' else nn.BatchNorm1d

        if num_layer1 > 0:
            self.net = nn.Sequential(*([nn.Linear(input_dim, mid_dim),
                                        Norm(mid_dim),
                                        nn.ReLU(True), ] +
                                       [nn.Linear(mid_dim, mid_dim),
                                        Norm(mid_dim),
                                        nn.ReLU(True), ] * (num_layer1 - 1)))
            input_dim = mid_dim

        if num_layer2 <= 1:
            self.kappa = nn.Sequential(nn.Linear(input_dim, 1),
                                       nn.Softplus(), )
            self.mean = nn.Sequential(nn.Linear(input_dim, mid_dim), )
        elif num_layer2 == 2:
            self.kappa = nn.Sequential(nn.Linear(input_dim, mid_dim),
                                       Norm(mid_dim),
                                       nn.ReLU(True),
                                       nn.Linear(mid_dim, 1),
                                       nn.Softplus(), )
            self.mean = nn.Sequential(nn.Linear(input_dim, mid_dim),
                                      Norm(mid_dim),
                                      nn.ReLU(True),
                                      nn.Linear(mid_dim, mid_dim), )
        else:
            self.kappa = nn.Sequential(*([nn.Linear(input_dim, mid_dim),
                                          Norm(mid_dim),
                                          nn.ReLU(True), ] +
                                         [nn.Dropout(0.5),
                                          nn.Linear(mid_dim, mid_dim),
                                          Norm(mid_dim),
                                          nn.ReLU(True), ] * (num_layer2 - 2) +
                                         [nn.Linear(mid_dim, 1),
                                          nn.Softplus(), ]))
            self.mean = nn.Sequential(*([nn.Linear(input_dim, mid_dim),
                                         Norm(mid_dim),
                                         nn.ReLU(True), ] +
                                        [nn.Dropout(0.5),
                                         nn.Linear(mid_dim, mid_dim),
                                         Norm(mid_dim),
                                         nn.ReLU(True), ] * (num_layer2 - 2) +
                                        [nn.Linear(mid_dim, mid_dim), ]))
        self.feature_dim = mid_dim
        self.beta = beta
        threshold = torch.tensor(20.)
        self.register_buffer('threshold', threshold)
        self.momentum = 1e-4

    def forward(self, x):
        if hasattr(self, 'net'):
            x = self.net(x)
        mean, kappa = self.mean(x), 1 / self.kappa(x)
        if not self.training:
            kappa[kappa > self.threshold] = (self.threshold * 1.5 - kappa[kappa > self.threshold]).clamp_min(1e-5)
            pass
        else:
            self.threshold = (1. - self.momentum) * self.threshold + self.momentum * kappa.max().item()
        return F.normalize(mean, 2, dim=-1), kappa  # * self.beta


