from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.layers.drop import DropPath

from src.layers import Bogomol

class BogomolBlock(nn.Module):
    def __init__(
        self, in_channels : int, hidden_dim : int, output_dim : int, block_len : int,
        kernel_size : Tuple[int], patch_size : Tuple[int], compressor_size : Tuple[int],
        padding : Tuple[int], patch_stride : Tuple[int], num_heads : int = 4
    ):
        super().__init__()
        self.bogomols = nn.ModuleList([
            Bogomol(
                in_channels, hidden_dim, in_channels,
                kernel_size, patch_size, compressor_size,
                padding, patch_stride, num_heads
            ) for _ in range(block_len)
        ])

        self.alpha = nn.Parameter(torch.ones(block_len))
        self.drop_path = DropPath(0.1)

        self.output_transform = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(
                in_channels, output_dim, 1,
                (2, 2)
            )
        )

    def forward(self, input : Tensor) -> Tensor:
        x = input
        for i, bogomol in enumerate(self.bogomols):
            x = x + self.alpha[i] * self.drop_path(bogomol(x))
        output = self.output_transform(x)
        return output

class ImageClassifier(nn.Module):
    def __init__(
        self, in_channels : int, hidden_dim : int, output_dim : int, entities : int,
        num_blocks : int = 2, block_len : int = 3, num_heads : int = 4,
        operating_size : Tuple[int] = (32, 32)
    ):
        super().__init__()
        self.kernel_size = (7, 7)
        self.patch_size = (8, 8)
        self.compressor_size = (7, 7)
        self.operating_size = operating_size

        self.padding = (3, 3)
        self.patch_stride = (3, 3)

        self.input_layer = Bogomol(
            in_channels, hidden_dim, entities,
            self.kernel_size, self.patch_size, self.compressor_size,
            self.padding, self.patch_stride, num_heads
        )

        self.bogomol_blocks = nn.ModuleList([
            BogomolBlock(
                entities*2**i, hidden_dim, entities*2**(i+1), block_len,
                self.kernel_size, self.patch_size, self.compressor_size,
                self.padding, self.patch_stride, num_heads
            ) for i in range(num_blocks)
        ])

        flat_shape = entities*2**num_blocks * (operating_size[0]//2**num_blocks) * (operating_size[1]//2**num_blocks)

        self.flatten = nn.Flatten()

        self.normalize = nn.LayerNorm(flat_shape)

        self.output = nn.Linear(flat_shape, output_dim)

    def forward(self, input : Tensor) -> Tensor:
        x = self.input_layer(input)
        x = F.interpolate(x, self.operating_size, mode='bicubic')
        for bogomol_block in self.bogomol_blocks:
            x = bogomol_block(x)
        x = self.flatten(x)
        x = self.normalize(x)
        result = self.output(x)
        return result

    def flops(self, input : Tensor) -> int:
        flops = 0


if __name__ == "__main__":
    import time
    batch_size = 5
    channels, height, width = 3, 32, 32
    tensor = torch.randn((batch_size, channels, height, width), requires_grad=True).to("cuda")
    hidden_dim = 64
    entities = 32
    num_classes = 10

    model = ImageClassifier(channels, hidden_dim, num_classes, entities, 3, 5, 8).to("cuda")
    time_start = time.time()
    y = model(tensor)
    print(y.shape)
    print(f"forward pass took {time.time()-time_start} second")
    time_start = time.time()
    loss = y.sum().backward()
    print(f"backward pass took {time.time()-time_start} second")
    print("Parameters count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
