from typing import Tuple
from math import log2

import torch
from torch import nn, Tensor

from src.layers import (
    ConvAttention1d, ConvAttention2d,
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
        self.k = torch.tensor([10])
        self.t = torch.tensor([0.1])

        self.patch_size = (int(log2(input_size[1])), int(log2(input_size[2])))
        self.patch_kernel = (self.patch_size[0]//2, self.patch_size[1]//2)

        self.normalization_layers = []
        self.conv_attention_layers = []

        self.conv_attention1 = ConvAttention2d(
            input_size[0], hidden_dim, input_size[1], input_size[2], self.patch_size,
            self.patch_kernel, self.output_kernel, self.padding, self.stride, self.patch_stride
        )

        output_size = self.__calculate_output_size((input_size[1], input_size[2]))

        for i in range(attention_blocks_count):
            self.normalization_layers.append(nn.LayerNorm(output_size))
            self.conv_attention_layers.append(
                BConvAttention2d(hidden_dim, hidden_dim, output_size[0], output_size[1],
                                 self.patch_size, self.patch_kernel, self.output_kernel,
                                 self.padding, self.stride, self.patch_stride)
            )
            output_size = self.__calculate_output_size(output_size, i != (attention_blocks_count-1))

        self.normalization_layers = nn.ModuleList(self.normalization_layers)
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

    def forward(self, input : Tensor, a : Tensor = torch.tensor([1.5])) -> Tensor:
        x = self.conv_attention1(input)
        for layer_norm, conv_attention in zip(self.normalization_layers, self.conv_attention_layers):
            x = layer_norm(x)
            x = conv_attention(x, a)
        x = self.layer_norm(x)
        x = self.flatten(x)
        result = self.linear1(x)
        return result

class ImageClassifier(nn.Module):
    def __init__(self, input_size : Tuple[int], hidden_dim : int, output_dim : int) -> None:
        super().__init__()
        assert input_size[1] > 1 and input_size[2] > 1
        self.output_kernel = (3, 3)
        self.padding = (3, 3)
        self.stride = (1, 1)
        self.patch_stride = (1, 1)

        self.patch_size = (int(log2(input_size[1])), int(log2(input_size[2])))
        self.patch_kernel = (self.patch_size[0]//2, self.patch_size[1]//2)

        self.conv_attention1 = ConvAttention2d(
            input_size[0], hidden_dim//4, input_size[1], input_size[2], self.patch_size,
            self.patch_kernel, self.output_kernel, self.padding, self.stride, self.patch_stride
        )

        output_size1 = self.__calculate_output_size((input_size[1], input_size[2]))

        self.layer_norm1 = nn.LayerNorm(output_size1)

        self.conv_attention2 = ConvAttention2d(
            hidden_dim//4, hidden_dim//2, output_size1[0], output_size1[1], self.patch_size,
            self.patch_kernel, self.output_kernel, self.padding, self.stride, self.patch_stride
        )

        output_size2 = self.__calculate_output_size(output_size1)

        self.layer_norm2 = nn.LayerNorm(output_size2)

        self.conv_attention3 = ConvAttention2d(
            hidden_dim//2, hidden_dim, output_size2[0], output_size2[1], self.patch_size,
            self.patch_kernel, self.output_kernel, self.padding, self.stride, self.patch_stride
        )

        output_size3 = self.__calculate_output_size(output_size2, False)

        self.layer_norm3 = nn.LayerNorm(output_size3)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(hidden_dim*output_size3[0]*output_size3[1], hidden_dim)

        self.activation = nn.GELU()

        self.layer_norm4 = nn.LayerNorm(hidden_dim)

        self.linear2 = nn.Linear(hidden_dim, output_dim)

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


    def forward(self, input : Tensor) -> Tensor:
        x = self.conv_attention1(input)
        x = self.layer_norm1(x)
        x = self.conv_attention2(x)
        x = self.layer_norm2(x)
        x = self.conv_attention3(x)
        x = self.layer_norm3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.layer_norm4(x)
        result = self.linear2(x)
        return result

if __name__ == "__main__":
    import time
    batch_size = 8
    channels, height, width = 3, 128, 128
    tensor = torch.randn((batch_size, channels, height, width), requires_grad=True).to("cuda")
    hidden_dim = 5
    num_classes = 10

    model = ImageClassifier((channels, height, width), hidden_dim, num_classes).to("cuda")
    time_start = time.time()
    y = model(tensor)
    print(y.shape)
    print(f"forward pass took {time.time()-time_start} second")
    time_start = time.time()
    loss = y.sum().backward()
    print(f"backward pass took {time.time()-time_start} second")
