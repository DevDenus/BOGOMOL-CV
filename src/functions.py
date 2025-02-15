from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Function

from src.utils.tensor_operations import (
    tensor_to_patches1d, tensor_to_patches2d,
    patches_to_tensor1d, patches_to_tensor2d
)

class SelfConv1d(Function):
    @staticmethod
    def forward(ctx, input : Tensor, patch_weights : Tensor, patch_size : int, patch_stride : int, padding : int = 0, stride : int = 1) -> Tensor:
        batch_size, channels, length = input.shape
        ch_out, num_patches, kernel_channels, kernel_length = patch_weights.shape
        assert length >= patch_size
        assert ch_out == channels
        assert num_patches == (length - patch_size) // patch_stride + 1
        assert kernel_channels == 1
        assert kernel_length <= patch_size

        ctx.patch_size = patch_size
        ctx.padding = padding
        ctx.stride = stride
        ctx.patch_stride = patch_stride

        patches = tensor_to_patches1d(input, patch_size, patch_stride)
        # [batch_size, num_patches, channels, patch_size]

        conv_patches = F.conv2d(patches, patch_weights)
        # [batch_size, ch_out, channels, patch_size']
        reshaped_input = input.reshape(1, batch_size*channels, length)
        reshaped_conv_patches = conv_patches.reshape(batch_size*ch_out, channels, -1)
        output = F.conv1d(reshaped_input, reshaped_conv_patches, padding=padding, stride=stride, groups=batch_size)
        # [1, batch_size*ch_out, length_out]
        output = output.reshape(batch_size, ch_out, -1)
        # [batch_size, ch_out, length_out]
        ctx.save_for_backward(input, patches, patch_weights, conv_patches)
        return output

    @staticmethod
    def backward(ctx, grad_output : Tensor) -> Tensor:
        patch_size = ctx.patch_size
        padding = ctx.padding
        stride = ctx.stride
        patch_stride = ctx.patch_stride
        input, patches, patch_weights, conv_patches = ctx.saved_tensors

        batch_size, channels, length = input.shape
        ch_out, num_patches, kernel_channels, kernel_length = patch_weights.shape

        output_padding1 = length - ((grad_output.shape[-1] - 1)*stride - 2 * padding + conv_patches.shape[-1])

        reshaped_input = input.reshape(1, batch_size*channels, length)
        reshaped_grad_output = grad_output.reshape(1, batch_size*ch_out, -1)
        reshaped_conv_patches = conv_patches.reshape(batch_size*ch_out, channels, -1)

        grad_input = F.conv_transpose1d(
            reshaped_grad_output, reshaped_conv_patches,
            stride=stride, padding=padding, groups=batch_size,
            output_padding=output_padding1
        ).reshape(batch_size, channels, length)
        grad_conv_patches = F.grad.conv1d_weight(
            reshaped_input, reshaped_conv_patches.shape, reshaped_grad_output,
            stride=stride, padding=padding, groups=batch_size
        ).reshape(batch_size, ch_out, channels, -1)

        output_padding2 = (0, patch_size - (grad_conv_patches.shape[-1] - 1 + kernel_length))

        grad_patches = F.conv_transpose2d(
            grad_conv_patches, patch_weights,
            output_padding=output_padding2
        )
        grad_patch_weights = F.grad.conv2d_weight(
            patches, patch_weights.shape, grad_conv_patches
        )

        grad_input += patches_to_tensor1d(grad_patches, length, patch_stride)
        return grad_input, grad_patch_weights, None, None, None, None


class SelfConv2d(Function):
    """
    Applies 2d-convolution of input tensor with given kernel, then applying 2d-convolution of the resulting tensor as a kernel to initial input tensor.
    """
    @staticmethod
    def forward(ctx, input : Tensor, patch_weights : Tensor, patch_size : Tuple[int, int], patch_stride : Tuple[int, int], padding : Tuple[int, int] = (0, 0), stride : Tuple[int, int] = (1, 1)) -> Tensor:
        batch_size, channels, height, width = input.shape
        ch_out, num_patches, kernel_channels, kernel_height, kernel_width = patch_weights.shape
        assert height >= patch_size[0]
        assert width >= patch_size[1]
        assert ch_out == channels
        assert num_patches == ((height - patch_size[0]) // patch_stride[0] + 1) * ((width - patch_size[1]) // patch_stride[1] + 1)
        assert kernel_channels == 1
        assert kernel_height <= patch_size[0]
        assert kernel_width <= patch_size[1]

        ctx.patch_size = patch_size
        ctx.padding = padding
        ctx.stride = stride
        ctx.patch_stride = patch_stride

        patches = tensor_to_patches2d(input, patch_size, patch_stride)
        # [batch_size, num_patches, channels, patch_height, patch_width]

        conv_patches = F.conv3d(patches, patch_weights)
        # [batch_size, ch_out, channels, patch_height', patch_width']
        reshaped_input = input.reshape(1, batch_size*channels, height, width)
        reshaped_conv_patches = conv_patches.reshape(batch_size*ch_out, channels, conv_patches.shape[-2], conv_patches.shape[-1])
        output = F.conv2d(reshaped_input, reshaped_conv_patches, padding=padding, stride=stride, groups=batch_size)
        # [1, batch_size*ch_out, height_out, width_out]
        output = output.reshape(batch_size, ch_out, output.shape[-2], output.shape[-1])
        # [batch_size, ch_out, height_out, width_out]
        ctx.save_for_backward(input, patches, patch_weights, conv_patches)
        return output

    @staticmethod
    def backward(ctx, grad_output : Tensor) -> Tensor:
        patch_size = ctx.patch_size
        padding = ctx.padding
        stride = ctx.stride
        patch_stride = ctx.patch_stride
        input, patches, patch_weights, conv_patches = ctx.saved_tensors

        batch_size, channels, height, width = input.shape
        ch_out, num_patches, kernel_channels, kernel_height, kernel_width = patch_weights.shape

        output_padding1 = (
            height - ((grad_output.shape[-2] - 1)*stride[0] - 2 * padding[0] + conv_patches.shape[-2]),
            width - ((grad_output.shape[-1] - 1)*stride[1] - 2 * padding[1] + conv_patches.shape[-1])
        )

        reshaped_input = input.view(1, batch_size*channels, height, width)
        reshaped_grad_output = grad_output.reshape(1, batch_size*ch_out, grad_output.shape[-2], grad_output.shape[-1])
        reshaped_conv_patches = conv_patches.reshape(batch_size*ch_out, channels, conv_patches.shape[-2], conv_patches.shape[-1])

        grad_input = F.conv_transpose2d(
            reshaped_grad_output, reshaped_conv_patches,
            stride=stride, padding=padding, groups=batch_size,
            output_padding=output_padding1
        ).reshape(batch_size, channels, height, width)
        grad_conv_patches = F.grad.conv2d_weight(
            reshaped_input, reshaped_conv_patches.shape, reshaped_grad_output,
            stride=stride, padding=padding, groups=batch_size
        ).reshape(batch_size, ch_out, channels, conv_patches.shape[-2], conv_patches.shape[-1])

        output_padding2 = (
            0,
            patch_size[0] - (grad_conv_patches.shape[-2] - 1 + kernel_height),
            patch_size[1] - (grad_conv_patches.shape[-1] - 1 + kernel_width)
        )

        grad_patches = F.conv_transpose3d(
            grad_conv_patches, patch_weights,
            output_padding=output_padding2
        )
        grad_patch_weights = F.grad.conv3d_weight(
            patches, patch_weights.shape, grad_conv_patches
        )

        grad_input += patches_to_tensor2d(grad_patches, (height, width), patch_stride)
        return grad_input, grad_patch_weights, None, None, None, None


# TODO: Develop this class after PyTorch release fold and unfold for 3D image-like output,
#       as well as F.conv4d
class SelfConv3d(Function):
    pass


if __name__ == "__main__":
    # Testing SelfConv1d

    batch_size, channels, length = 4, 3, 128
    patch_size = 3
    padding = 7
    stride = 5
    patch_stride = 3
    tensor = torch.randn((batch_size, channels, length), requires_grad=True).to(float)
    num_patches = (length - patch_size) // patch_stride + 1
    patch_weight = torch.randn((channels, num_patches, 1, 3), requires_grad=True).to(float)
    self_conv1d = SelfConv1d().apply(tensor, patch_weight, patch_size, patch_stride, padding, stride)
    loss = self_conv1d.sum().backward()
    test = torch.autograd.gradcheck(SelfConv1d.apply, (tensor, patch_weight, patch_size, patch_stride, padding, stride))
    print(test)

    batch_size, channels, height, width = 4, 3, 48, 48
    patch_size = (5, 5)
    padding = (0, 0)
    stride = (1, 1)
    patch_stride = patch_size
    tensor = torch.randn((batch_size, channels, height, width), requires_grad=True).to(float)
    num_patches = ((height - patch_size[0]) // patch_stride[0] + 1) * ((width - patch_size[1]) // patch_stride[1] + 1)
    patch_weight = torch.randn((channels, num_patches, 1, 3, 3), requires_grad=True).to(float)
    self_conv2d = SelfConv2d().apply(tensor, patch_weight, patch_size, patch_stride, padding, stride)
    print(self_conv2d.shape)
    loss = self_conv2d.sum().backward()
    test = torch.autograd.gradcheck(SelfConv2d.apply, (tensor, patch_weight, patch_size, patch_stride, padding, stride))
    print(test)
