from math import log2
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Bogomol(nn.Module):
    def __init__(
        self, input_dim : int, emb_dim : int, output_dim : int,
        input_size : Tuple[int, int], patch_size : Tuple[int, int],
        kernel_size : Tuple[int, int], patch_stride : Tuple[int, int]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.patch_dim = input_dim * self.patch_size[0] * self.patch_size[1]
        self.num_patches = ((input_size[0] - self.patch_size[0]) // self.patch_stride[0] + 1) * \
            ((input_size[1] - self.patch_size[1]) // self.patch_stride[1] + 1)

        self.form_query = nn.Sequential(
            nn.Linear(self.patch_dim, self.emb_dim, bias=False),
            nn.LayerNorm(self.emb_dim)
        )

        self.form_key = nn.Sequential(
            nn.Linear(self.patch_dim, self.emb_dim, bias=False),
            nn.LayerNorm(self.emb_dim)
        )

        self.activation = nn.GELU()

        self.patch_pos_emb = nn.Parameter(torch.empty(self.num_patches, self.emb_dim))
        nn.init.xavier_normal_(self.patch_pos_emb)

        self.alpha = nn.Parameter(
            torch.tensor((self.num_patches)**(-0.5))
        )
        self.temp_dim = int(input_dim**0.5)

        self.height_decompressor = nn.Linear(self.emb_dim, self.temp_dim*input_size[0])
        self.width_decompressor = nn.Linear(self.emb_dim, self.temp_dim*input_size[1])

        padding = (
            kernel_size[0] // 2,
            kernel_size[1] // 2
        )

        self.output_proj = nn.Conv2d(
            self.temp_dim**2, output_dim, kernel_size,
            padding=padding, bias=False
        )

    def forward(self, input : Tensor) -> Tensor:
        batch_size, channels, height, width = input.shape
        patches = F.unfold(input, self.patch_size, stride=self.patch_stride).transpose(1, 2).contiguous()
        flat_patches = patches.view(
            batch_size, self.num_patches, self.patch_dim
        )
        query = self.activation(self.form_query(flat_patches) + self.patch_pos_emb)
        key = self.activation(self.form_key(flat_patches) + self.patch_pos_emb)
        compressed_context = torch.bmm(query.transpose(1, 2), key) * torch.abs(self.alpha)
        context = self.width_decompressor(self.height_decompressor(compressed_context).transpose(1, 2)).view(
            batch_size, self.temp_dim, height, self.temp_dim, width
        ).transpose(2, 3).contiguous().view(
            batch_size, self.temp_dim**2, height, width
        )
        output = self.output_proj(context)
        return output


if __name__ == "__main__":
    import time
    batch_size = 4
    channels = 32
    height, width = 32, 32
    tensor = torch.randn(batch_size, channels, height, width, requires_grad=True).to('cuda')
    out_channels = 32

    kernel_size = (3, 3)
    patch_size = (7, 7)
    stride = (1, 1)
    padding = (3, 3)
    patch_stride = (3, 3)

    hidden_dim = 32

    model2d = Bogomol(
        channels, hidden_dim, out_channels,
        (height, width), patch_size,
        kernel_size, patch_stride
    ).to('cuda')
    #print(f"MFLOPs: {model2d.flops(tensor.shape)/1e+6:.1f}")
    forward_min_time = float('inf')
    backward_min_time = float('inf')
    for i in range(10):
        tensor = torch.randn(batch_size, channels, height, width, requires_grad=True).to('cuda')
        time_start = time.time()
        y = model2d(tensor)
        forward_min_time = min(forward_min_time, time.time()-time_start)
        time_start = time.time()
        loss = y.sum().backward()
        backward_min_time = min(backward_min_time, time.time()-time_start)
    #model2d.eval()
    #with torch.no_grad():
    #    tensor = torch.randn(batch_size, channels, height, width, requires_grad=True, dtype=float).to('cuda')
    #    y = model2d(tensor)
    #    y_sample = model2d(tensor[0].unsqueeze(0))
    #    print(f"Not mixing samples: {abs(y[0] - y_sample).max()}")
    print(f"forward pass took {forward_min_time} second")
    print(f"backward pass took {backward_min_time} second")
