from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

def tensor_to_patches1d(tensor : Tensor, patch_size : int, stride : int) -> Tensor:
    """
    Splits batch of vectors into the fixed-sized patches.

    input:
        tensor (Tensor): Tensor of size [batch_size, channels, length].
        patch_length (int): Patch length.
        stride (int, optional): stride between patches, if None - stride = patch_length.

    returns:
        Tensor: Tensor of patches [batch_size, num_patches, channels, patch_length],
            where num_patches = (length - patch_length) // stride + 1.
    """
    assert stride is not None

    batch_size, channels, length = tensor.shape

    patches = tensor.unfold(dimension=2, size=patch_size, step=stride)
    # [batch_size, channels, num_patches, patch_length]
    patches = patches.permute(0, 2, 1, 3)
    # [batch_size, num_patches, channels, patch_length]
    return patches

def patches_to_tensor1d(patches : Tensor, original_size : int, stride : int) -> Tensor:
    """
    The inverse operation of tensor_to_patches1d.
    Used to restore gradient of input from gradient of patches.

    input:
        patches (Tensor): Tensor of size [batch_size, num_patches, channels, patch_length].
        original_size (int): Input length.
        stride (int, optional): stride between patches, if None - stride = patch_length.

    returns:
        Tensor: Input tensor of patches [batch_size, channels, length]
    """
    assert stride is not None

    batch_size, num_patches, channels, patch_length = patches.shape

    patches = patches.permute(0, 2, 3, 1)
    # [batch_size, channels, patch_length, num_patches]
    tensor = patches.reshape(batch_size, channels * patch_length, num_patches)

    tensor = F.fold(
        tensor, output_size=(original_size,1),
        kernel_size=(patch_length,1), stride=(stride,1)
    )
    # [batch_size, channels, original_size]
    return tensor.squeeze(-1)


def tensor_to_patches2d(tensor : Tensor, patch_size: Tuple[int, int], stride : Tuple[int, int]) -> Tensor:
    """
    Splits the batch of images into fixed-sized patches.

    input:
        tensor (Tensor): Tensor of shape [batch_size, channels, height, width].
        patch_size (tuple): Patch size (patch_height, patch_width).
        stride (tuple, optional): stride between patches (stride_height, stride_width).
            If None, then stride = patch_size.

    return:
        Tensor: Tensor of patches [batch_size, num_patches, channels, patch_height, patch_width],
            where num_patches = ((height - patch_height) // stride_height + 1) * ((width - patch_width) // stride_width + 1).
    """
    assert stride is not None

    batch_size, channels, height, width = tensor.shape
    patch_height, patch_width = patch_size

    patches = F.unfold(tensor, kernel_size=(patch_height, patch_width), stride=stride)
    # [batch_size, channels * patch_height * patch_width, num_patches]
    patches = patches.reshape(batch_size, channels, patch_height, patch_width, -1)
    # [batch_size, channels, patch_height, patch_width, num_patches]
    patches = patches.permute(0, 4, 1, 2, 3)
    # [batch_size, num_patches, channels, patch_height, patch_width]
    return patches

def patches_to_tensor2d(patches : Tensor, original_size: Tuple[int, int], stride : Tuple[int, int]) -> Tensor:
    """
    The inverse operation of tensor_to_patches2d.
    Used to restore gradient of input from gradient of patches.

    input:
        patches (Tensor): Tensor of size [batch_size, num_patches, channels, patch_height, patch_width].
        original_size (tuple): Input size.
        stride (tuple, optional): stride between patches, if None - stride = patch_length.

    returns:
        Tensor: Input tensor of patches [batch_size, channels, height, width]
    """
    assert stride is not None

    batch_size, num_patches, channels, patch_height, patch_width = patches.shape

    patches = patches.permute(0, 2, 3, 4, 1)
    # [batch_size, channels, patch_height, patch_width, num_patches]
    tensor = patches.reshape(batch_size, channels * patch_height * patch_width, num_patches)

    tensor = F.fold(
        tensor, output_size=original_size,
        kernel_size=(patch_height, patch_width), stride=stride
    )
    # [batch_size, channels, original_size[0], original_size[1]]

    return tensor.squeeze(-1)

# TODO: Develop after PyTorch release torch.nn.Unfold for 3D image like tensors
def tensor_to_patches3d(tensor : Tensor, patch_size : Tuple[int, int, int], stride : Optional[Tuple[int, int, int]] = None) -> Tensor:
    """
    Splits batch of 3D images into fixed-sized patches of 3D images.

    input:
        tensor (Tensor): Tensor of 3D images [batch_size, channels, depth, height, width].
        patch_size (tuple): Patch size (patch_depth, patch_height, patch_width).
        stride (tuple, optional): stride between patches (stride_depth, stride_height, stride_width).
            If None, then stride = (patch_depth, patch_height, patch_width).

    return:
        Tensor: Tensor of patches [batch_size, num_patches, channels, patch_depth, patch_height, patch_width],
            where num_patches = ((depth - patch_depth) // stride_depth + 1) *
            ((height - patch_height) // stride_height + 1) * ((width - patch_width) // stride_width + 1).
    """
    pass

# TODO: Develop after PyTorch release torch.nn.Fold for 3D image like tensors
def patches_to_tensor3d(patches : Tensor, original_size : Tuple[int, int, int], stride : Optional[Tuple[int, int, int]] = None) -> Tensor:
    """
    The inverse operation of tensor_to_patches3d.
    Used to restore gradient of input from gradient of patches.

    input:
        patches (Tensor): Tensor of size [batch_size, num_patches, channels, patch_depth, patch_height, patch_width].
        original_size (tuple): Input shape.
        stride (tuple, optional): stride between patches, if None - stride = (patch_depth, patch_height, patch_width).

    returns:
        Tensor: Input tensor of patches [batch_size, channels, depth, height, width]
    """
    pass

if __name__ == '__main__':
    batch_size = 1
    channels = 3
    length = 10
    patch_length = 5
    stride_vector = 5
    vectors = torch.randn(batch_size, channels, length)
    vector_patches = tensor_to_patches1d(vectors, patch_length, stride=stride_vector)
    restored_vector = patches_to_tensor1d(vector_patches, length, stride_vector)
    print("Vector batch:")
    print("Initial shape:", vectors.shape)
    print("Patches shape:", vector_patches.shape)
    print("Restored shape:", restored_vector.shape)
    print("Restored successful:", (abs(vectors-restored_vector) < 1e-6).all())

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
