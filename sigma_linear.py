# based on "Stabilizing Transformer Training by Preventing Attention Entropy Collapse"
# https://arxiv.org/abs/2303.06296
# modified from original 

# usage - replace your current nn.Linear with SigmaLinear

import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmaLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer('u', F.normalize(torch.randn(in_features), dim=0))
        with torch.no_grad():
            sigma =self.get_sigma()
        self.register_buffer('spectral_norm', sigma)
        self.sigma = nn.Parameter(torch.ones(1))
    
    def get_sigma(self):
        with torch.no_grad():
            u = self.u  
            v = self.weight.mv(u)
            v = F.normalize(v, dim=0)
            u = self.weight.T.mv(v)
            u = F.normalize(u, dim=0)
            self.u.data.copy_(u)
        result = torch.sum(v * torch.matmul(self.weight, u))
    
    def get_weight(self,):
        sigma = self.get_sigma()
        if self.training:
            self.spectral_norm.data.copy_(sigma)
        weight = (self.sigma/sigma) * self.weight
        return weight
    
    def forward(self, x):
        return F.linear(x, self.get_weight(), self.bias)
