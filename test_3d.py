import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import DeformConv

num_deformable_groups = 2

N, inC, inH, inW, inD = 2, 6, 128, 128, 128
outC, outH, outW, outD = 4, 128, 128, 128
kH, kW, kD = 3, 3, 3

conv = nn.Conv3d(
    inC,
    num_deformable_groups * 3 * kH * kW * kD,
    kernel_size=(kH, kW, kD),
    stride=(1, 1, 1),
    padding=(1, 1, 1),
    bias=False,
).cuda()

conv_offset3d = DeformConv(
    3,
    inC,
    outC,
    (kH, kW, kD),
    stride=1,
    padding=1,
    num_deformable_groups=num_deformable_groups,
).cuda()

inputs = Variable(torch.randn(N, inC, inH, inW, inD).cuda(), requires_grad=True)
offset = conv(inputs)
output = conv_offset3d(inputs, offset)
output.backward(output.data)
print(output.size())
