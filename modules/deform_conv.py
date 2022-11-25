import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair, _triple
from functions import deform_conv_function


class DeformConv(Module):
    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 num_deformable_groups=1):
        super(DeformConv, self).__init__()
        self.dim = dim
        make_list = _triple if dim == 3 else _pair
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_list(kernel_size)
        self.stride = make_list(stride)
        self.padding = make_list(padding)
        self.dilation = make_list(dilation)
        self.num_deformable_groups = num_deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return deform_conv_function(self.dim, input, offset, self.weight, self.stride,
                             self.padding, self.dilation,
                             self.num_deformable_groups)
