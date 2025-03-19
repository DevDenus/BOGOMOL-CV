from typing import Tuple
from math import log2

import torch
from torch import nn, Tensor

from src.layers import (
    SelfConv2d,
    BConvAttention2d
)

class BImageClassifier(nn.Module):
    def __init__(self, input_size : Tuple[int], hidden_dim : int, output_dim : int, attention_blocks_count : int = 10) -> None:
        super().__init__()
        assert input_size[1] > 1 and input_size[2] > 1
        self.output_kernel = (3, 3)
        self.padding = (1, 1)
        self.stride = (1, 1)
        self.patch_stride = (1, 1)

        self.patch_size = (7, 7)
        self.patch_kernel = (3, 3)

        self.conv_attention_layers = []
        self.layer_norm_layers = []

        self.conv_attention1 = ConvAttention2d(
            input_size[0], hidden_dim, input_size[1], input_size[2], self.patch_size,
            self.patch_kernel, self.output_kernel, self.padding, self.stride, self.patch_stride
        )

        output_size = self.__calculate_output_size((input_size[1], input_size[2]))

        for _ in range(attention_blocks_count):
            self.layer_norm_layers.append(nn.LayerNorm(output_size))
            self.conv_attention_layers.append(
                BConvAttention2d(hidden_dim, hidden_dim, output_size[0], output_size[1],
                                 self.patch_size, self.patch_kernel, self.output_kernel,
                                 self.padding, self.stride, self.patch_stride)
            )
            output_size = self.__calculate_output_size(output_size, False)

        self.layer_norm_layers = nn.ModuleList(self.layer_norm_layers)
        self.conv_attention_layers = nn.ModuleList(self.conv_attention_layers)

        self.layer_norm = nn.LayerNorm(output_size)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(hidden_dim*output_size[0]*output_size[1], output_dim)

    def __calculate_output_size(self, input_size : Tuple[int, int], update_patch_params : bool = True) -> Tuple[int, int]:
        conv_patch_size = (
            (self.patch_size[0] - self.patch_kernel[0])//self.patch_stride[0] + 1,
            (self.patch_size[1] - self.patch_kernel[1])//self.patch_stride[1] + 1
        )
        conv_attn_size = (
            (input_size[0] - conv_patch_size[0] + 2*self.padding[0])//self.stride[0] + 1,
            (input_size[1] - conv_patch_size[1] + 2*self.padding[1])//self.stride[1] + 1
        )
        output_size = (
            (conv_attn_size[0] - self.output_kernel[0] + 2*self.padding[0])//self.stride[0] + 1,
            (conv_attn_size[1] - self.output_kernel[1] + 2*self.padding[1])//self.stride[1] + 1
        )
        if update_patch_params:
            self.patch_size = (int(log2(output_size[0])), int(log2(output_size[1])))
            self.patch_kernel = (self.patch_size[0]//2, self.patch_size[1]//2)
        return output_size

    # def forward(self, input : Tensor, k : Tensor = torch.tensor([1]), t : Tensor = torch.tensor([30])) -> Tensor:
    def forward(self, input : Tensor, a : Tensor = torch.tensor([500])) -> Tensor:
        x = self.conv_attention1(input)
        for conv_attention, layer_norm in zip(self.conv_attention_layers, self.layer_norm_layers):
            x = layer_norm(x)
            # x = conv_attention(x, k, t)
            x = conv_attention(x, a) + x
        x = self.layer_norm(x)
        x = self.flatten(x)
        result = self.linear1(x)
        return result

class MantisBlock(nn.Module):
    def __init__(self, input_dim : int, hidden_dim : int, output_dim : int) -> None:
        super().__init__()
        self.patch_size = (6, 6)
        self.compressor_size = (3, 3)
        self.kernel_size = (3, 3)

        self.padding = (1, 1)
        self.stride = (1, 1)
        self.patch_stride = (3, 3)

        self.instance_norm1 = nn.InstanceNorm2d(input_dim, eps=1e-3)

        self.conv = nn.Conv2d(
            input_dim, hidden_dim, self.kernel_size,
            self.stride, self.padding
        )

        self.instance_norm2 = nn.InstanceNorm2d(hidden_dim, eps=1e-3)

        self.self_conv = SelfConv2d(
            hidden_dim, output_dim, self.patch_size, self.compressor_size,
            self.padding, self.stride, self.patch_stride
        )

        self.activation = nn.ReLU()

        # self.scaling_factor = nn.Parameter(0.5*torch.ones(1, output_dim, 1, 1))



    def forward(self, input : Tensor) -> Tensor:
        x = self.instance_norm1(input)
        x = self.conv(x)
        x = self.activation(x)
        x = self.instance_norm2(x)
        x = self.self_conv(x)
        x = self.activation(x)
        return x

class ImageClassifier(nn.Module):
    def __init__(self, input_size : Tuple[int], hidden_dim : int, output_dim : int, attention_blocks_count : int = 5) -> None:
        super().__init__()
        self.patch_size = (6, 6)
        self.compressor_size = (3, 3)
        self.kernel_size = (3, 3)

        self.padding = (1, 1)
        self.stride = (1, 1)
        self.patch_stride = (3, 3)

        self.input = MantisBlock(input_size[0], hidden_dim//2, hidden_dim)

        output_size = self.__calculate_output_size((input_size[1], input_size[2]))

        self.mantis_blocks = []

        for i in range(attention_blocks_count):
            self.mantis_blocks.append(
                MantisBlock(hidden_dim, hidden_dim//2, hidden_dim)
            )
            output_size = self.__calculate_output_size(output_size)

        self.mantis_blocks = nn.ModuleList(self.mantis_blocks)

        self.flatten = nn.Flatten()

        self.layer_norm = nn.LayerNorm(hidden_dim*output_size[0]*output_size[1], eps=1e-3)

        #self.linear = nn.Linear(hidden_dim*output_size[0]*output_size[1], hidden_dim)

        #self.activation = nn.GELU()

        self.output = nn.Linear(hidden_dim*output_size[0]*output_size[1], output_dim)

    def __calculate_output_size(self, input_size : Tuple[int, int]) -> Tuple[int, int]:
        conv_size = (
            (input_size[0] - self.kernel_size[0] + 2*self.padding[0])//self.stride[0] + 1,
            (input_size[1] - self.kernel_size[1] + 2*self.padding[1])//self.stride[1] + 1
        )
        compressed_patch_size = (
            (self.patch_size[0] - self.compressor_size[0]) + 1,
            (self.patch_size[1] - self.compressor_size[1]) + 1
        )
        self_conv_size = (
            (conv_size[0] - compressed_patch_size[0] + 2*self.padding[0])//self.stride[0] + 1,
            (conv_size[1] - compressed_patch_size[1] + 2*self.padding[1])//self.stride[1] + 1
        )
        return self_conv_size


    def forward(self, input : Tensor) -> Tensor:
        x = self.input(input)
        for mantis in self.mantis_blocks:
            x = mantis(x)
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
    hidden_dim = 16
    num_classes = 10

    model = ImageClassifier((channels, height, width), hidden_dim, num_classes).to("cuda")
    time_start = time.time()
    y = model(tensor)
    print(y.shape)
    print(f"forward pass took {time.time()-time_start} second")
    time_start = time.time()
    loss = y.sum().backward()
    print(f"backward pass took {time.time()-time_start} second")
