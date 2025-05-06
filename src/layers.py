from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class FilterFormer(nn.Module):
    def __init__(
        self, input_dim : int, hidden_dim : int, output_dim : int,
        input_size : Tuple[int], target_size : Tuple[int], num_heads : int = 4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.target_size = target_size
        self.target_dim = output_dim*target_size[0]*target_size[1]
        self.image_dim = input_size[0]*input_size[1]

        self.input_projector = nn.Sequential(
            nn.Linear(self.image_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)

        self.alpha = nn.Parameter(torch.tensor(.5))

        self.target_projector = nn.Linear(hidden_dim, self.target_dim, bias=False)


    def forward(self, input : Tensor) -> Tensor:
        batch_size, channels, height, width = input.shape
        input_flat = input.view(
            batch_size, channels, self.image_dim
        )
        target_proj = self.input_projector(input_flat)
        attentive_targets, _ = self.attention(target_proj, target_proj, target_proj)
        filter_proj = self.target_projector(attentive_targets + self.alpha*target_proj)
        dynamic_filters = filter_proj.view(
            batch_size, self.input_dim, self.output_dim, *self.target_size
        ).transpose(1, 2).contiguous()
        return dynamic_filters


class Bogomol(nn.Module):
    def __init__(
        self, input_dim : int, output_dim : int, hidden_dim : int, input_size : Tuple[int],
        kernel_size : Tuple[int], dynamic_filter_size : Tuple[int], padding : Tuple[int] = (0, 0),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.dynamic_filter_size = dynamic_filter_size
        self.padding = padding
        self.features_size = (
            self.input_size[0] - self.kernel_size[0] + 2*self.padding[0] + 1,
            self.input_size[1] - self.kernel_size[1] + 2*self.padding[1] + 1
        )

        self.input_layer = nn.Conv2d(
            input_dim, input_dim, self.kernel_size,
            padding=padding, bias=False
        )

        self.filter_former = FilterFormer(
            input_dim, hidden_dim, output_dim,
            self.features_size, dynamic_filter_size
        )


    def forward(self, input : Tensor) -> Tensor:
        input_features = self.input_layer(input)
        batch_size, _, feature_height, feature_width = input_features.shape

        dynamic_filters = self.filter_former(input_features).view(
            batch_size*self.output_dim, self.input_dim, *self.dynamic_filter_size
        )

        deformed_input = input_features.view(
            1, batch_size*self.input_dim, feature_height, feature_width
        )

        self_conv = F.conv2d(
            deformed_input, dynamic_filters, padding=self.padding, groups=batch_size
        )
        _, _, self_conv_height, self_conv_width = self_conv.shape
        self_conv = self_conv.view(
            batch_size, self.output_dim, self_conv_height, self_conv_width
        )
        return self_conv

    def flops(self, input : Tensor) -> int:
        flops = 0
        return flops


if __name__ == "__main__":
    import time
    batch_size = 4
    channels = 32
    height, width = 32, 32
    tensor = torch.randn(batch_size, channels, height, width, requires_grad=True).to('cuda')
    out_channels = 64

    kernel_size = (3, 3)
    filter_size = (7, 7)
    stride = (1, 1)
    padding = (2, 2)

    hidden_dim = 48

    model2d = Bogomol(
        channels, out_channels, hidden_dim,
        (height, width), kernel_size, filter_size,
        padding
    ).to('cuda')
    print(f"MFLOPs: {model2d.flops(tensor.shape)/1e+6:.1f}")
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
