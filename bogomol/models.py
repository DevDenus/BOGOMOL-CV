from math import sqrt
from typing import Tuple

import torch
from torch import nn, Tensor
from timm.layers.drop import DropPath

from bogomol.layers import Bogomol


class BogomolBlock(nn.Module):
    def __init__(
        self, in_channels : int, output_dim : int, block_len : int,
        input_size : Tuple[int], kernel_size : Tuple[int], patch_stride_ratio : float
    ):
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.block_len = block_len
        self.patch_size = (
            int(sqrt(input_size[0])),
            int(sqrt(input_size[1]))
        )
        self.patch_stride = (
            max(int(self.patch_size[0]*patch_stride_ratio), 1),
            max(int(self.patch_size[1]*patch_stride_ratio), 1)
        )
        self.hidden_dim = in_channels*(self.patch_size[0] + self.patch_size[1])//2

        self.bogomols = nn.ModuleList([
            nn.Sequential(
                Bogomol(
                    in_channels, self.hidden_dim, in_channels,
                    input_size, self.patch_size,
                    kernel_size, self.patch_stride
                ),
                nn.GroupNorm(4, in_channels)
            ) for _ in range(block_len)
        ])

        self.alpha = nn.Parameter(torch.ones(block_len)/block_len)
        self.drop_path = DropPath(0.1)

        padding = (
            kernel_size[0] // 2,
            kernel_size[1] // 2
        )

        self.output_transform = nn.Sequential(
            nn.Conv2d(
                in_channels, output_dim, kernel_size,
                stride=(2, 2), padding=padding, bias=False
            ),
            nn.GroupNorm(4, output_dim),
            nn.GELU()
        )

    def forward(self, input : Tensor) -> Tensor:
        x = input
        for i, bogomol in enumerate(self.bogomols):
            x = x + torch.abs(self.alpha[i]) * self.drop_path(bogomol(x))
        output = self.output_transform(x)
        return output

    def flops(self):
        bogomol_flops = 0
        for bogomol in self.bogomols:
            bogomol_flops += bogomol[0].flops() + 9 * self.in_channels * self.input_size[0] * self.input_size[1]
        conv_flops = 2*self.in_channels * self.output_dim * self.kernel_size[0] * self.kernel_size[1] * self.input_size[0] // 2 * self.input_size[1] // 2
        output_flops = conv_flops + 15*self.output_dim * self.input_size[0] // 2 * self.input_size[1] // 2
        return bogomol_flops + output_flops

class ImageClassifier(nn.Module):
    def __init__(
        self, in_channels : int, hidden_dim : int, output_dim : int,
        num_blocks : int = 3, block_len : int = 5,
        image_size : Tuple[int] = (32, 32)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.image_size = image_size
        self.num_blocks = num_blocks
        self.kernel_size = (5, 5)
        self.patch_stride_ratio = 0.5

        padding = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2
        )

        self.input_layer = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_dim, self.kernel_size,
                bias=False, padding=padding
            ),
            nn.GroupNorm(4, hidden_dim)
        )

        self.bogomol_blocks = nn.ModuleList([
            BogomolBlock(
                hidden_dim*2**i, hidden_dim*2**(i+1), block_len,
                (self.image_size[0]//(2**i), self.image_size[1]//(2**i)),
                self.kernel_size, self.patch_stride_ratio
            ) for i in range(num_blocks)
        ])

        self.flat_size = hidden_dim * 2**num_blocks

        self.avg_flat = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.output = nn.Linear(self.flat_size, output_dim)

    def forward(self, input : Tensor) -> Tensor:
        x = self.input_layer(input)
        for bogomol_block in self.bogomol_blocks:
            x = bogomol_block(x)
        x_flat = self.avg_flat(x)
        result = self.output(x_flat)
        return result

    def flops(self):
        conv_flops = 2 * self.in_channels*self.hidden_dim * self.image_size[0] * self.image_size[1] * self.kernel_size[0] * self.kernel_size[1] + 9 * self.hidden_dim * self.image_size[0] * self.image_size[1]
        bogomol_block_flops = 0
        for bogomol_block in self.bogomol_blocks:
            bogomol_block_flops += bogomol_block.flops()
        avg_pool_flops = self.hidden_dim * 2**self.num_blocks * self.image_size[0] // 2**(self.num_blocks-1) * self.image_size[1] // 2**(self.num_blocks-1)
        output_flops = 2 * self.flat_size * self.output_dim
        total_flops = conv_flops + bogomol_block_flops + avg_pool_flops + output_flops
        return total_flops

if __name__ == "__main__":
    import time
    batch_size = 1
    channels, height, width = 3, 64, 64
    tensor = torch.randn((batch_size, channels, height, width), requires_grad=True).to("cuda")
    entities = 64
    num_classes = 10

    model = ImageClassifier(channels, entities, num_classes, 4, 5, (height, width)).to("cuda")
    torch.compile(model)
    forward_min_time = float('inf')
    backward_min_time = float('inf')
    for i in range(10):
        tensor = torch.randn(batch_size, channels, height, width, requires_grad=True).to('cuda')
        time_start = time.time()
        y = model(tensor)
        forward_min_time = min(forward_min_time, time.time()-time_start)
        time_start = time.time()
        loss = y.sum().backward()
        backward_min_time = min(backward_min_time, time.time()-time_start)
    print(f"forward pass took {forward_min_time} second")
    print(f"backward pass took {backward_min_time} second")
    print("Parameters count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"MFLOPs: {model.flops()/1e+6:.1f}")
