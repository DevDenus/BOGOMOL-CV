import math
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

def get_coords(tensor_size : Tuple[int], patch_size : Tuple[int, int], stride : Tuple[int, int]) -> Tensor:
    batch_size, channels, height, width = tensor_size
    num_patches_height = (height - patch_size[0]) // stride[0] + 1
    num_patches_width = (width - patch_size[1]) // stride[1] + 1

    patch_cords_x = torch.arange(num_patches_width) * stride[1] + patch_size[1] / 2
    patch_cords_y = torch.arange(num_patches_height) * stride[0] + patch_size[0] / 2

    grid_x, grid_y = torch.meshgrid(patch_cords_x, patch_cords_y, indexing='xy')

    x_norm = grid_x / width
    y_norm = grid_y / height

    coords = torch.stack([
        x_norm,
        y_norm
    ], dim=-1).view(
        -1, 2
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    coords.requires_grad = False
    return coords


def tensor_to_patches2d(tensor : Tensor, patch_size: Tuple[int, int], stride : Tuple[int, int]) -> Tuple[Tensor, Tensor]:
    """
    Splits the batch of images into fixed-sized patches.

    input:
        tensor (Tensor): Tensor of shape [batch_size, channels, height, width].
        patch_size (tuple): Patch size (patch_height, patch_width).
        stride (tuple): stride between patches (stride_height, stride_width).

    return:
        Tensor: Tensor of patches [batch_size, num_patches, channels, patch_height, patch_width],
            where num_patches = ((height - patch_height) // stride_height + 1) * ((width - patch_width) // stride_width + 1).
    """
    assert stride is not None

    batch_size, channels, _, _ = tensor.shape
    patch_height, patch_width = patch_size

    patches = F.unfold(tensor, kernel_size=(patch_height, patch_width), stride=stride)
    patches = patches.view(batch_size, channels, patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 1, 2, 3).contiguous()

    patches_coords = get_coords(tensor.shape, patch_size, stride).to(tensor.device)

    return patches, patches_coords

if __name__ == '__main__':
    batch_size = 8
    channels = 3
    height, width = 32, 32
    patch_size_img = (8, 8)
    stride_img = (3, 3)
    images = torch.randn(batch_size, channels, height, width)
    image_patches, patch_coords = tensor_to_patches2d(images, patch_size_img, stride=stride_img)
    print("\nImage batch:")
    print("Initial shape:", images.shape)
    print("Patches shape:", image_patches.shape)
    print(patch_coords.shape)
