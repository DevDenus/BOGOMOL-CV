import torch

from typing import Any
from torch import nn

from IRNet_complete.binary_functions import (
    BinaryQuantize, BMatrixMul, ZhegalkinTransform1d,
    ZhegalkinTransform2d, BConv2d
)

class ZhegalkinLinear(nn.Linear):
    """
    Analogue of nn.Linear with binary matrix multiplication instead of continuos and
    using both inputs and their pairwise conjunctions, providing theoretical completeness.
    Applying Libre-PB quantization on input according to IR-Net study.
    Using sign as activation in forward pass and sign approximation according to IR-Net
    study in backward pass.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device = None, dtype = None):
        """
        Weight matrix would be of size (out_features, in_features*(in_features-1)/2) as
        ZhegalkinTransformation is applied.
        """
        super().__init__(in_features=in_features*(in_features-1)//2, out_features=out_features,
                         bias=bias, device=device, dtype=dtype)
        self.k = torch.tensor([10]).float()
        self.t = torch.tensor([0.1]).float()

    def forward(self, input):
        x = ZhegalkinTransform1d().apply(input)
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(
            torch.tensor([2]*bw.size(0)).cuda().float(),
            (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / torch.log(2)).round().float()
            ).view(bw.size(0), 1, 1, 1).detach()
        bw = BinaryQuantize().apply(bw, self.k, self.t)
        bx = BinaryQuantize().apply(x, self.k, self.t)
        bw = bw * sw
        out = BMatrixMul().apply(bx, bw, self.bias)
        return out

class ZhegalkinConv2d(nn.Conv2d):
    """
    Analogue of nn.Linear with binary matrix multiplication instead of continuos and
    using both inputs and their pairwise conjunctions, providing theoretical completeness.
    Applying Libre-PB quantization on input according to IR-Net study.
    Using sign as activation in forward pass and sign approximation according to IR-Net
    study in backward pass.
    """
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int = 1,
                 padding : int = 0, dilation : int = 0, groups : int = 1, bias : bool = True,
                 padding_mode : str = 'zeros', device : Any = None, dtype : Any = None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                         device=device, dtype=dtype)
        self.k = torch.tensor([10]).float()
        self.t = torch.tensor([0.1]).float()

    def forward(self, input):
        x = ZhegalkinTransform2d().apply(input)
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(
            torch.tensor([2]*bw.size(0)).cuda().float(),
            (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / torch.log(2)).round().float()
            ).view(bw.size(0), 1, 1, 1).detach()
        bw = BinaryQuantize().apply(bw, self.k, self.t)
        bx = BinaryQuantize().apply(x, self.k, self.t)
        bw = bw * sw
        out = BConv2d().apply(bx, bw, self.bias)
        return out
