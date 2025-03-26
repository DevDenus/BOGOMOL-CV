from typing import Tuple, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import src.functions as func


class SelfConv2d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, patch_size : Tuple[int, int], compressor_size : Tuple[int, int], padding : Tuple[int, int] = (0, 0), stride : Tuple[int, int] = (1, 1), patch_stride : Optional[Tuple[int, int]] = None):
        super().__init__()
        if patch_stride is None:
            patch_stride = patch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.compressor_size = compressor_size
        self.patch_stride = patch_stride
        self.padding = padding
        self.stride = stride

        self.patch_weight = nn.Parameter(
            torch.empty(self.in_channels, self.patch_size[0], self.patch_size[1])
        )
        self.patch_bias = nn.Parameter(
            torch.empty(self.in_channels, self.patch_size[0], self.patch_size[1])
        )
        self.compressor = nn.Parameter(
            torch.empty(self.out_channels, self.compressor_size[0], self.compressor_size[1])
        )

        nn.init.xavier_uniform_(self.patch_weight)
        nn.init.constant_(self.patch_bias, 0.)
        nn.init.xavier_uniform_(self.compressor)


    def forward(self, input : Tensor) -> Tensor:
        assert input.shape[-2] >= self.patch_size[0], \
            f"Can't form patches of form {self.patch_size} out of input of form {(input.shape[-2], input.shape[-2])}"
        assert input.shape[-1] >= self.patch_size[1], \
            f"Can't form patches of form {self.patch_size} out of input of form {(input.shape[-2], input.shape[-2])}"
        self_attention = func.SelfConv2d.apply(
            input, self.patch_weight, self.patch_bias, self.compressor,
            self.patch_size, self.patch_stride, self.padding, self.stride
        )
        return self_attention

class BConvAttention2d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, max_input_height : int, max_input_width : int,
                 patch_size : Tuple[int, int], patch_kernel : Tuple[int, int], output_kernel : Tuple[int, int],
                 padding : Tuple[int, int] = (0, 0), stride : Tuple[int, int] = (1, 1),
                 patch_stride : Optional[Tuple[int, int]] = None):
        super().__init__()
        if patch_stride is None:
            patch_stride = patch_size

        self.quantizer = Linear2SignEde

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

    # def forward(self, input : Tensor, k : Tensor, t : Tensor) -> Tensor:
    def forward(self, input : Tensor, a : Tensor) -> Tensor:
        assert input.shape[-2] <= self.max_input_height
        assert input.shape[-1] <= self.max_input_width
        binput = self.quantizer.apply(input, a)
        bpatch_filters = self.quantizer.apply(self.patch_filters, a)
        padded_binput = F.pad(binput,
            (0, self.max_input_height - input.shape[-2], 0, self.max_input_width - input.shape[-1])
        )
        self_attention = SelfConv2d.apply(
            padded_binput, bpatch_filters,
            self.patch_size, self.patch_stride, self.padding, self.stride
        )
        bself_attention = self.quantizer.apply(self_attention, a)
        boutput_filters = self.quantizer.apply(self.output_filters, a)
        output = F.conv2d(
            bself_attention, boutput_filters,
            stride=self.stride, padding=self.padding
        )
        return output


if __name__ == "__main__":
    import time

    batch_size = 32
    channels = 3
    height, width = 32, 32
    tensor = torch.randn(batch_size, channels, height, width, requires_grad=True)
    out_channels = 5
    max_height, max_width = 32, 32
    patch_size = (5, 5)
    compressor_size = (3, 3)

    model2d = SelfConv2d(channels, out_channels, patch_size, compressor_size)
    time_start = time.time()
    y = model2d(tensor)
    print(y.shape)
    print(f"forward pass took {time.time()-time_start} second")
    time_start = time.time()
    loss = y.sum().backward()
    print(f"backward pass took {time.time()-time_start} second")
