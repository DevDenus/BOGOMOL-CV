from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


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

    batch_size, channels, height, width = tensor.shape
    patch_height, patch_width = patch_size

    patches = F.unfold(tensor, kernel_size=(patch_height, patch_width), stride=stride)
    # [batch_size, channels * patch_height * patch_width, num_patches]
    patches = patches.view(batch_size, channels, patch_height, patch_width, -1)
    # [batch_size, channels, patch_height, patch_width, num_patches]
    patches = patches.permute(0, 4, 1, 2, 3).contiguous()
    # [batch_size, num_patches, channels, patch_height, patch_width]

    return patches

def patches_to_tensor2d(patches : Tensor, original_size: Tuple[int, int], stride : Tuple[int, int]) -> Tensor:
    """
    The inverse operation of tensor_to_patches2d.
    Used to restore gradient of input from gradient of patches.

    input:
        patches (Tensor): Tensor of size [batch_size, num_patches, channels, patch_height, patch_width].
        original_size (tuple): Input size.
        stride (tuple): stride between patches.

    returns:
        Tensor: Input tensor of patches [batch_size, channels, height, width]
    """
    assert stride is not None

    batch_size, num_patches, channels, patch_height, patch_width = patches.shape

    patches = patches.permute(0, 2, 3, 4, 1).contiguous()
    # [batch_size, channels, patch_height, patch_width, num_patches]
    tensor = patches.view(batch_size, channels * patch_height * patch_width, num_patches)

    tensor = F.fold(
        tensor, output_size=original_size,
        kernel_size=(patch_height, patch_width), stride=stride
    ).squeeze(-1)
    # [batch_size, channels, original_size[0], original_size[1]]

    return tensor

if __name__ == '__main__':
    batch_size = 8
    channels = 3
    height, width = 128, 128
    patch_size_img = (32, 32)
    stride_img = (16, 16)
    images = torch.randn(batch_size, channels, height, width)
    image_patches = tensor_to_patches2d(images, patch_size_img, stride=stride_img)
    restored_image = patches_to_tensor2d(image_patches, (height, width), stride_img)
    print("\nImage batch:")
    print("Initial shape:", images.shape)
    print("Patches shape:", image_patches.shape)
    print("Restored shape:", restored_image.shape)
    print("Restored successful:", (abs(images-restored_image) < 1e-6).all())
