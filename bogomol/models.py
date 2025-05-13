from math import sqrt, log2
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
        self.block_len = block_len
        self.patch_size = (
            int(sqrt(input_size[0])),
            int(sqrt(input_size[1]))
        )
        self.patch_stride = (
            max(int(self.patch_size[0]*patch_stride_ratio), 1),
            max(int(self.patch_size[1]*patch_stride_ratio), 1)
        )
        self.hidden_dim = int(sqrt(in_channels*self.patch_size[0]*self.patch_size[1]))

        self.bogomols = nn.ModuleList([
            nn.Sequential(
                Bogomol(
                    in_channels, self.hidden_dim, in_channels,
                    input_size, self.patch_size,
                    kernel_size, self.patch_stride
                ),
                nn.GroupNorm(8, in_channels),
                nn.GELU()
            ) for _ in range(block_len)
        ])

        self.alpha = nn.Parameter(torch.ones(block_len)/block_len)
        self.drop_path = DropPath(0.05)

        self.output_transform = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                in_channels, output_dim, (1, 1), bias=False
            ),
            nn.GroupNorm(8, output_dim)
        )


    def forward(self, input : Tensor) -> Tensor:
        x = input
        for i, bogomol in enumerate(self.bogomols):
            x = x + torch.abs(self.alpha[i]) * self.drop_path(bogomol(x))
        output = self.output_transform(x)
        return output

class ImageClassifier(nn.Module):
    def __init__(
        self, in_channels : int, hidden_dim : int, output_dim : int,
        num_blocks : int = 3, block_len : int = 5, image_size : Tuple[int] = (32, 32)
    ):
        super().__init__()
        self.image_size = image_size
        self.kernel_size = (5, 5)
        self.patch_stride_ratio = 1.

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

        self.flat_size = hidden_dim*2**num_blocks * \
            self.image_size[0]//(2**num_blocks) * \
            self.image_size[1]//(2**num_blocks)

        self.avg_flat = nn.Flatten()

        self.output = nn.Linear(self.flat_size, output_dim)

    def forward(self, input : Tensor) -> Tensor:
        x = self.input_layer(input)
        for bogomol_block in self.bogomol_blocks:
            x = bogomol_block(x)
        x_flat = self.avg_flat(x)
        result = self.output(x_flat)
        return result



if __name__ == "__main__":
    import time
    batch_size = 64
    channels, height, width = 3, 32, 32
    tensor = torch.randn((batch_size, channels, height, width), requires_grad=True).to("cuda")
    entities = 32
    num_classes = 10

    model = ImageClassifier(channels, entities, num_classes, 3, 5, (height, width)).to("cuda")
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
