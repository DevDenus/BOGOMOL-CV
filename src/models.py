from typing import Tuple

import torch
from torch import nn, Tensor

from src.layers import SelfConv2d

class BogomolBlock(nn.Module):
    def __init__(
        self, input_dim : int, hidden_dim : int, output_dim : int, blocks_count : int,
        kernel_size : Tuple[int, int], patch_size : Tuple[int, int], compressor_size : Tuple[int, int],
        padding : Tuple[int, int], stride : Tuple[int, int], patch_stride : Tuple[int, int]
    ) -> None:
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(
                input_dim, hidden_dim, kernel_size,
                stride, padding
            ),
            nn.InstanceNorm2d(hidden_dim, eps=1e-3),
        )

        self.blocks = []
        for _ in range(blocks_count):
            block = nn.Sequential(
                SelfConv2d(
                    hidden_dim, hidden_dim, patch_size, compressor_size,
                    padding, stride, patch_stride
                ),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim, hidden_dim, kernel_size,
                    stride, padding
                ),
                nn.InstanceNorm2d(hidden_dim, eps=1e-3),
            )
            self.blocks.append(block)

        self.blocks = nn.ModuleList(self.blocks)


        self.output = nn.Sequential(
            SelfConv2d(
                hidden_dim, output_dim, patch_size, compressor_size,
                padding, stride, patch_stride
            ),
            nn.GELU(),
            nn.MaxPool2d((2, 2))
        )

        # self.scaling_factor = nn.Parameter(0.5*torch.ones(1, output_dim, 1, 1))

    def forward(self, input : Tensor) -> Tensor:
        x = self.input_layer(input)
        for block in self.blocks:
            x = block(x) + x
        output = self.output(x)
        return output

class ImageClassifier(nn.Module):
    def __init__(self, input_size : Tuple[int], hidden_dim : int, output_dim : int, block_len : int = 5, blocks_count : int = 2) -> None:
        super().__init__()
        self.kernel_size = (5, 5)
        self.patch_size = (7, 7)
        self.compressor_size = (3, 3)

        self.padding = (2, 2)
        self.stride = (1, 1)
        self.patch_stride = (3, 3)

        self.input_layer = BogomolBlock(input_size[0], hidden_dim, hidden_dim, block_len, self.kernel_size, self.patch_size, self.compressor_size, self.padding, self.stride, self.patch_stride)

        output_size = self.__calculate_output_size((input_size[1], input_size[2]))

        self.bogomol_blocks = []

        for i in range(blocks_count):
            self.bogomol_blocks.append(
                BogomolBlock(
                    hidden_dim*2**i, hidden_dim*2**(i+1), hidden_dim*2**(i+1),
                    block_len, self.kernel_size, self.patch_size, self.compressor_size,
                    self.padding, self.stride, self.patch_stride
                )
            )
            output_size = self.__calculate_output_size(output_size)

        self.bogomol_blocks = nn.ModuleList(self.bogomol_blocks)

        self.flatten = nn.Flatten()

        self.layer_norm = nn.LayerNorm(hidden_dim*(2**blocks_count)*output_size[0]*output_size[1], eps=1e-3)

        self.output = nn.Linear(hidden_dim*(2**blocks_count)*output_size[0]*output_size[1], output_dim)

    def __calculate_output_size(self, input_size : Tuple[int, int]) -> Tuple[int, int]:
        conv_size = (
            (input_size[0] - self.kernel_size[0] + 2*self.padding[0])//self.stride[0] + 1,
            (input_size[1] - self.kernel_size[1] + 2*self.padding[1])//self.stride[1] + 1
        )
        compressed_patch_size = (
            self.patch_size[0] - self.compressor_size[0] + 1,
            self.patch_size[1] - self.compressor_size[1] + 1
        )
        self_conv_size = (
            (conv_size[0] - compressed_patch_size[0] + 2*self.padding[0])//self.stride[0] + 1,
            (conv_size[1] - compressed_patch_size[1] + 2*self.padding[1])//self.stride[1] + 1
        )
        max_pool_size = (
            self_conv_size[0] // 2,
            self_conv_size[1] // 2
        )
        return max_pool_size


    def forward(self, input : Tensor) -> Tensor:
        x = self.input_layer(input)
        for bogomol in self.bogomol_blocks:
            x = bogomol(x)
        x = self.flatten(x)
        #x = self.layer_norm(x)
        #x = self.linear(x)
        #x = self.activation(x)
        x = self.layer_norm(x)
        result = self.output(x)
        return result

if __name__ == "__main__":
    import time
    batch_size = 8
    channels, height, width = 3, 32, 32
    tensor = torch.randn((batch_size, channels, height, width), requires_grad=True).to("cuda")
    hidden_dim = 32
    num_classes = 10

    model = ImageClassifier((channels, height, width), hidden_dim, num_classes).to("cuda")
    time_start = time.time()
    y = model(tensor)
    print(y.shape)
    print(f"forward pass took {time.time()-time_start} second")
    time_start = time.time()
    loss = y.sum().backward()
    print(f"backward pass took {time.time()-time_start} second")
    print("Parameters count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
