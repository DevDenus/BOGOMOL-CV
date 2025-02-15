from typing import Tuple, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from src.functions import SelfConv1d, SelfConv2d
from src.quant.ops import BinarySignEdeFn

class ConvAttention1d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, max_input_length : int,
                 patch_size : int, patch_kernel : int, output_kernel : int, padding : int = 0,
                 stride : int = 1, patch_stride : Optional[int] = None):
        super().__init__()
        if patch_stride is None:
            patch_stride = patch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_input_length = max_input_length
        self.patch_size = patch_size
        self.patch_kernel = patch_kernel
        self.padding = padding
        self.stride = stride
        self.patch_stride = patch_stride

        self.num_patches = (max_input_length - patch_size) // patch_stride + 1

        self.patch_filters = nn.Parameter(
            torch.empty(self.in_channels, self.num_patches, 1, self.patch_kernel)
        )
        self.output_filters = nn.Parameter(
            torch.empty(out_channels, in_channels, output_kernel)
        )

        self.activation = nn.SELU()

    def forward(self, input : Tensor) -> Tensor:
        assert input.shape[-1] <= self.max_input_length
        padded_input = F.pad(input, (0, self.max_input_length - input.shape[-1]))
        self_attention = SelfConv1d.apply(
            padded_input, self.patch_filters,
            self.patch_size, self.patch_stride, self.padding, self.stride
        )
        output = F.conv1d(
            self_attention, self.output_filters,
            stride=self.stride, padding=self.padding
        )
        output = self.activation(output)
        return output

class ConvAttention2d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, max_input_height : int, max_input_width : int,
                 patch_size : Tuple[int, int], patch_kernel : Tuple[int, int], output_kernel : Tuple[int, int],
                 padding : Tuple[int, int] = (0, 0), stride : Tuple[int, int] = (1, 1),
                 patch_stride : Optional[Tuple[int, int]] = None):
        super().__init__()
        if patch_stride is None:
            patch_stride = patch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_input_height = max_input_height
        self.max_input_width = max_input_width
        self.patch_size = patch_size
        self.patch_kernel = patch_kernel
        self.padding = padding
        self.stride = stride
        self.patch_stride = patch_stride

        self.num_patches = ((max_input_height - patch_size[0]) // patch_stride[0] + 1) * ((max_input_width - patch_size[1]) // patch_stride[1] + 1)

        self.patch_filters = nn.Parameter(
            torch.empty(self.in_channels, self.num_patches, 1, self.patch_kernel[0], self.patch_kernel[1])
        )

        self.output_filters = nn.Parameter(
            torch.empty(out_channels, in_channels, output_kernel[0], output_kernel[1])
        )

        nn.init.kaiming_normal_(self.patch_filters, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.output_filters, mode='fan_in', nonlinearity='relu')

        self.activation = nn.GELU()

    def forward(self, input : Tensor) -> Tensor:
        assert input.shape[-2] <= self.max_input_height
        assert input.shape[-1] <= self.max_input_width
        padded_input = F.pad(input,
            (0, self.max_input_height - input.shape[-2], 0, self.max_input_width - input.shape[-1])
        )
        self_attention = SelfConv2d.apply(
            padded_input, self.patch_filters,
            self.patch_size, self.patch_stride, self.padding, self.stride
        )
        output = F.conv2d(
            self_attention, self.output_filters,
            stride=self.stride, padding=self.padding
        )
        output = self.activation(output)
        return output

class BConvAttention2d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, max_input_height : int, max_input_width : int,
                 patch_size : Tuple[int, int], patch_kernel : Tuple[int, int], output_kernel : Tuple[int, int],
                 padding : Tuple[int, int] = (0, 0), stride : Tuple[int, int] = (1, 1),
                 patch_stride : Optional[Tuple[int, int]] = None):
        super().__init__()
        if patch_stride is None:
            patch_stride = patch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_input_height = max_input_height
        self.max_input_width = max_input_width
        self.patch_size = patch_size
        self.patch_kernel = patch_kernel
        self.padding = padding
        self.stride = stride
        self.patch_stride = patch_stride

        self.num_patches = ((max_input_height - patch_size[0]) // patch_stride[0] + 1) * ((max_input_width - patch_size[1]) // patch_stride[1] + 1)

        self.patch_filters = nn.Parameter(
            torch.empty(self.in_channels, self.num_patches, 1, self.patch_kernel[0], self.patch_kernel[1])
        )

        self.output_filters = nn.Parameter(
            torch.empty(out_channels, in_channels, output_kernel[0], output_kernel[1])
        )

        nn.init.xavier_uniform_(self.patch_filters)
        nn.init.xavier_uniform_(self.output_filters)

    def forward(self, input : Tensor, k : Tensor, t : Tensor) -> Tensor:
        assert input.shape[-2] <= self.max_input_height
        assert input.shape[-1] <= self.max_input_width
        binput = BinarySignEdeFn.apply(input, k, t)
        bpatch_filters = BinarySignEdeFn.apply(self.patch_filters, k, t)
        padded_binput = F.pad(binput,
            (0, self.max_input_height - input.shape[-2], 0, self.max_input_width - input.shape[-1])
        )
        self_attention = SelfConv2d.apply(
            padded_binput, bpatch_filters,
            self.patch_size, self.patch_stride, self.padding, self.stride
        )
        bself_attention = BinarySignEdeFn.apply(self_attention, k, t)
        boutput_filters = BinarySignEdeFn.apply(self.output_filters, k, t)
        output = F.conv2d(
            bself_attention, boutput_filters,
            stride=self.stride, padding=self.padding
        )
        return output


if __name__ == "__main__":
    import time

    batch_size = 4
    channels = 3
    length = 128
    tensor = torch.randn(batch_size, channels, length, requires_grad=True)
    out_channels = 5
    max_length = 128
    patch_size = 8
    patch_kernel = 3
    output_kernel = 5

    print("1d test")
    model1d = ConvAttention1d(channels, out_channels, max_length, patch_size, patch_kernel, output_kernel, 5)
    time_start = time.time()
    y = model1d(tensor)
    print(y.shape)
    print(f"forward pass took {time.time()-time_start} second")
    time_start = time.time()
    loss = y.sum().backward()
    print(tensor.grad.shape)
    print(f"backward pass took {time.time()-time_start} second")


    batch_size = 32
    channels = 3
    height, width = 32, 32
    tensor = torch.randn(batch_size, channels, height, width, requires_grad=True)
    out_channels = 5
    max_height, max_width = 32, 32
    patch_size = (5, 5)
    patch_kernel = (3, 3)
    output_kernel = (3, 3)

    print("2d test")
    model2d = BConvAttention2d(channels, out_channels, max_height, max_width, patch_size, patch_kernel, output_kernel)
    time_start = time.time()
    y = model2d(tensor)
    print(y.shape)
    print(f"forward pass took {time.time()-time_start} second")
    time_start = time.time()
    loss = y.sum().backward()
    print(f"backward pass took {time.time()-time_start} second")
