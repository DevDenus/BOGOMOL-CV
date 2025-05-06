from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.layers.drop import DropPath

from src.layers import Bogomol

class BogomolBlock(nn.Module):
    def __init__(
        self, in_channels : int, output_dim : int, hidden_dim : int, block_len : int,
        input_size : Tuple[int], kernel_size : Tuple[int], dynamic_filter_size : Tuple[int],
        padding : Tuple[int]
    ):
        super().__init__()
        self.bogomols = nn.ModuleList([
            nn.Sequential(
                Bogomol(
                    in_channels, in_channels, hidden_dim,
                    input_size, kernel_size, dynamic_filter_size,
                    padding=padding,
                ),
                nn.GroupNorm(1, in_channels),
                nn.GELU()
            ) for _ in range(block_len)
        ])

        self.alpha = nn.Parameter(torch.zeros(block_len))
        self.drop_path = DropPath(0.1)

        self.output_transform = nn.Sequential(
            nn.Conv2d(
                in_channels, output_dim, 1,
                (2, 2), bias=False
            ),
            nn.GroupNorm(1, output_dim),
            nn.GELU()
        )

    def forward(self, input : Tensor) -> Tensor:
        x = input
        alphas = F.softmax(self.alpha, dim=0)
        for i, bogomol in enumerate(self.bogomols):
            x = x + alphas[i] * self.drop_path(bogomol(x))
        output = self.output_transform(x)
        return output

class ImageClassifier(nn.Module):
    def __init__(
        self, in_channels : int, hidden_dim : int, entities : int, output_dim : int,
        num_blocks : int = 2, block_len : int = 4, image_size : Tuple[int] = (32, 32), num_sights : int = 8
    ):
        super().__init__()
        self.kernel_size = (3, 3)
        self.dynamic_filter_size = (7, 7)
        self.image_size = image_size

        self.padding = (2, 2)
        self.input_padding = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2
        )

        self.input_layer = nn.Sequential(
            nn.Conv2d(
                in_channels, entities, self.kernel_size, padding=self.input_padding
            ),
            nn.GroupNorm(1, entities)
        )

        self.bogomol_blocks = nn.ModuleList([
            BogomolBlock(
                entities*2**i, entities*2**(i+1), hidden_dim, block_len,
                (self.image_size[0]//(2**i), self.image_size[1]//(2**i)),
                self.kernel_size, self.dynamic_filter_size, self.padding
            ) for i in range(num_blocks)
        ])

        self.flatten = nn.Flatten()

        flat_size = entities*2**num_blocks * \
            self.image_size[0]//(2**num_blocks)* \
            self.image_size[1]//(2**num_blocks)

        self.output = nn.Linear(flat_size, output_dim)

    def forward(self, input : Tensor) -> Tensor:
        x = F.interpolate(input, self.image_size, mode='bicubic')
        x = self.input_layer(x)
        for bogomol_block in self.bogomol_blocks:
            x = bogomol_block(x)
        x_flat = self.flatten(x)
        result = self.output(x_flat)
        return result

    def flops(self, input : Tensor) -> int:
        flops = 0


if __name__ == "__main__":
    import time
    batch_size = 64
    channels, height, width = 3, 32, 32
    tensor = torch.randn((batch_size, channels, height, width), requires_grad=True).to("cuda")
    hidden_dim = 64
    entities = 64
    num_classes = 10

    model = ImageClassifier(channels, hidden_dim, entities, num_classes, 3, 5, num_sights=32).to("cuda")
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
