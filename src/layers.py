from typing import Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from src.utils.tensor_operations import tensor_to_patches2d

class SE(nn.Module):
    def __init__(self, in_channels : int, bottle_neck_coeff : int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//bottle_neck_coeff, 1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//bottle_neck_coeff, in_channels, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class Bogomol(nn.Module):
    def __init__(
        self, input_dim : int, hidden_dim : int, output_dim : int, kernel_size : Tuple[int],
        patch_size : Tuple[int], compressor_size : Tuple[int], padding : Tuple[int] = (0, 0),
        patch_stride : Tuple[int] = None, num_heads : int = 4
    ):
        super().__init__()
        if patch_stride is None:
            patch_stride = patch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.compressor_size = compressor_size

        self.patch_dim = output_dim * patch_size[0] * patch_size[1]
        self.compressor_dim = output_dim * compressor_size[0] * compressor_size[1]

        self.padding = padding
        self.patch_stride = patch_stride

        self.input_layer = nn.Sequential(
            nn.GroupNorm(1, input_dim),
            nn.Conv2d(
                input_dim, output_dim, kernel_size,
                padding=padding
            ),
            nn.GELU()
        )

        self.position_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh()
        )

        self.compress = nn.Sequential(
            nn.Linear(self.patch_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.patch_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.non_linearity = nn.GELU()

        self.basis_election = nn.Sequential(
            nn.Linear(hidden_dim, compressor_size[0]*compressor_size[1]),
            nn.Tanh()
        )

        self.patch_basis = nn.Parameter(
            torch.empty(compressor_size[0]*compressor_size[1], self.compressor_dim)
        )
        nn.init.xavier_uniform_(self.patch_basis)

        self.representation_election = nn.Sequential(
            nn.Linear(hidden_dim, self.compressor_dim),
            nn.GELU()
        )

        self.squeeze_excitation = SE(output_dim)
        self.alpha = nn.Parameter(torch.tensor(.5))

    def forward(self, input : Tensor) -> Tensor:
        batch_size = input.shape[0]
        input_features = self.input_layer(input)
        _, _, feature_height, feature_width = input_features.shape

        patches, patches_coords = tensor_to_patches2d(input_features, self.patch_size, self.patch_stride)

        seq_length = patches.shape[1]

        flat_patches = patches.view(
            batch_size, seq_length, -1
        )
        patch_emb = self.compress(flat_patches)
        pos_emb = self.position_embedding(patches_coords)
        patch_pos_emb = patch_emb + pos_emb
        patch_attentive, _ = self.patch_attention(patch_pos_emb, patch_pos_emb, patch_pos_emb)
        patch_attentive = self.non_linearity(patch_attentive)

        patch_votes = self.basis_election(patch_attentive).mean(1).unsqueeze(-1)
        basis_decision = patch_votes * self.patch_basis.unsqueeze(0).expand(batch_size, -1, -1)

        #patches_representatives = self.form_representations(patches_decision.transpose(1, 2)).transpose(1, 2).contiguous()
        represent_decision = self.representation_election(patch_attentive).mean(1).view(
            batch_size, self.output_dim, self.compressor_size[0]*self.compressor_size[1]
        )
        patches_representatives = torch.bmm(represent_decision, basis_decision)

        dynamic_filters = patches_representatives.view(
            batch_size*self.output_dim, self.output_dim, self.compressor_size[0], self.compressor_size[1]
        )

        reshaped_input = input_features.view(
            1, batch_size*self.output_dim, feature_height, feature_width
        )
        self_conv = F.conv2d(
            reshaped_input, dynamic_filters,
            padding=self.padding, groups=batch_size
        )
        _, _, selfconv_height, selfconv_width = self_conv.shape
        self_conv = self_conv.view(
            batch_size, self.output_dim, selfconv_height, selfconv_width
        )
        return self.squeeze_excitation(self.alpha*input_features + self_conv)

    def flops(self, input : Tensor) -> int:
        flops = 0
        return flops


if __name__ == "__main__":
    import time

    batch_size = 4
    channels = 32
    height, width = 224, 224
    tensor = torch.randn(batch_size, channels, height, width, requires_grad=True).to('cuda')
    out_channels = 64

    kernel_size = (5, 5)
    patch_size = (32, 32)
    compressor_size = (5, 5)
    patch_stride = (16, 16)
    padding = (2, 2)

    hidden_dim = 32

    model2d = Bogomol(
        channels, hidden_dim, out_channels,
        kernel_size, patch_size, compressor_size,
        padding, patch_stride
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
    # y_sample = model2d(tensor[0].unsqueeze(0))
    # print(f"Not mixing samples: {(y[0] == y_sample).all()}")
    print(f"forward pass took {forward_min_time} second")
    print(f"backward pass took {backward_min_time} second")
