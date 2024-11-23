import torch
import numpy as np
import torch.fft as fourier
import torch.nn as nn
import math
import os
import torch.nn.functional as F
class PCC(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, O, G):
        # O = O.reshape(O.shape[0], -1)
        # G = G.reshape(G.shape[0], -1)
        mean_O = torch.mean(O, dim=1, keepdim=True)
        mean_G = torch.mean(G, dim=1, keepdim=True)
        # print(mean_G.size())
        numerator = torch.sum((O - mean_O) * (G - mean_G), dim=1)
        denominator = torch.sqrt(torch.sum(torch.square(O - mean_O), dim=1, keepdim=True) * torch.sum(torch.square(G - mean_G), dim=1, keepdim=True))
        # print(denominator.size())
        # sum or average? the same!
        numerator = torch.sum(numerator)
        denominator = torch.sum(denominator)
        return numerator / denominator