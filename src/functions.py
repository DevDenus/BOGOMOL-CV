from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

from src.utils.tensor_operations import (
    tensor_to_patches2d,
    patches_to_tensor2d
)

class SelfConv2d(Function):
    """
    Applies 2d-convolution of input tensor with given kernel, then applying 2d-convolution of the resulting tensor as a kernel to initial input tensor.
    """
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    def forward(ctx, input : Tensor, filter_weights : Tensor, filter_bias : Tensor, compressor : Tensor, patch_size : Tuple[int, int], patch_stride : Tuple[int, int], padding : Tuple[int, int] = (0, 0), stride : Tuple[int, int] = (1, 1)) -> Tensor:
        batch_size, channels, height, width = input.shape
        weight_channels, weight_height, weight_width = filter_weights.shape
        bias_channels, bias_height, bias_width = filter_bias.shape
        entities,  comp_height, comp_width = compressor.shape
        assert height >= patch_size[0]
        assert width >= patch_size[1]
        assert weight_channels == channels
        assert weight_height == patch_size[0]
        assert weight_width == patch_size[1]
        assert bias_channels == channels
        assert bias_height == patch_size[0]
        assert bias_width == patch_size[1]
        assert comp_height <= patch_size[0]
        assert comp_width <= patch_size[1]

        ctx.batch_size = batch_size
        ctx.channels = channels
        ctx.patch_size = patch_size
        ctx.padding = padding
        ctx.stride = stride
        ctx.patch_stride = patch_stride

        # Forming patches
        patches = tensor_to_patches2d(input, patch_size, patch_stride)
        # [batch_size, num_patches, channels, patch_height, patch_width]
        num_patches = patches.shape[-4]

        # Performing filtration
        reshaped_patches = patches.view(batch_size*num_patches, channels, patch_size[0], patch_size[1])
        biased_patches = reshaped_patches + filter_bias
        filtered_patches = torch.relu(biased_patches)
        weighted_filtered_patches = filter_weights * filtered_patches

        # Compressing patches to a Tensor of specified shape
        reshaped_weighted_filtered_patches = weighted_filtered_patches.view(
            batch_size, num_patches, channels, patch_size[0], patch_size[1]
        ).transpose(1, 2).contiguous().view(
            batch_size*channels, num_patches, patch_size[0], patch_size[1]
        )
        compressor_kernel = compressor.unsqueeze(0).expand(num_patches, -1, -1, -1).transpose(0, 1).contiguous()
        compressed_patches = F.conv2d(reshaped_weighted_filtered_patches, compressor_kernel)
        # [batch_size*channels, entities, patch_height', patch_width']

        # Convolving input tensor with compressed patches
        _, _, compressed_height, compressed_width = compressed_patches.shape
        reshaped_compressed_patches = compressed_patches.view(
            batch_size, channels, entities, compressed_height, compressed_width
        ).transpose(1, 2).contiguous().view(
            batch_size*entities, channels, compressed_height, compressed_width
        ).contiguous()
        reshaped_input = input.reshape(1, batch_size*channels, height, width)
        output = F.conv2d(reshaped_input, reshaped_compressed_patches, padding=padding, stride=stride, groups=batch_size)
        # [1, batch_size*out_channels, height_out, width_out]
        output = output.view(batch_size, entities, output.shape[-2], output.shape[-1]).contiguous()

        ctx.save_for_backward(
            reshaped_input, reshaped_compressed_patches, reshaped_weighted_filtered_patches, compressor_kernel, filtered_patches, filter_weights
        )
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output : Tensor) -> Tensor:
        batch_size = ctx.batch_size
        channels = ctx.channels
        patch_size = ctx.patch_size
        padding = ctx.padding
        stride = ctx.stride
        patch_stride = ctx.patch_stride
        reshaped_input, reshaped_compressed_patches, reshaped_weighted_filtered_patches, compressor_kernel, filtered_patches, filter_weights = ctx.saved_tensors
        _, _, height, width = reshaped_input.shape
        entities, num_patches, comp_height, comp_width = compressor_kernel.shape
        #print("num_patches", num_patches)
        #print("grad_output", grad_output)

        _, _, grad_output_height, grad_output_width = grad_output.shape
        reshaped_grad_output = grad_output.reshape(1, batch_size*entities, grad_output_height, grad_output_width)
        _, _, comp_patches_height, comp_patches_width = reshaped_compressed_patches.shape

        output_padding1 = (
            height - ((grad_output_height - 1)*stride[0] - 2 * padding[0] + comp_patches_height),
            width - ((grad_output_width - 1)*stride[1] - 2 * padding[1] + comp_patches_width)
        )

        grad_input = F.conv_transpose2d(
            reshaped_grad_output, reshaped_compressed_patches,
            stride=stride, padding=padding, groups=batch_size,
            output_padding=output_padding1
        ).view(batch_size, channels, height, width)
        grad_compressed_patches = F.grad.conv2d_weight(
            reshaped_input, reshaped_compressed_patches.shape, reshaped_grad_output,
            stride=stride, padding=padding, groups=batch_size
        ).view(
            batch_size, entities, channels, comp_patches_height, comp_patches_width
        ).transpose(1, 2).contiguous().view(
            batch_size*channels, entities, comp_patches_height, comp_patches_width
        )
        #print("grad_input", grad_input)
        #print("grad_compressed_patches", grad_compressed_patches)

        grad_weighted_filtered_patches = F.conv_transpose2d(
            grad_compressed_patches, compressor_kernel
        ).view(
            batch_size, channels, num_patches, patch_size[0], patch_size[1]
        ).transpose(1, 2).contiguous().view(
            batch_size*num_patches, channels, patch_size[0], patch_size[1]
        )
        grad_compressor = F.grad.conv2d_weight(
            reshaped_weighted_filtered_patches, compressor_kernel.shape, grad_compressed_patches
        ).sum(1)
        #print("grad_weighted_filtered_patches", grad_weighted_filtered_patches)
        #print("grad_compressor", grad_compressor)

        grad_filtered_patches = filter_weights * (filtered_patches > 0).to(filtered_patches.dtype) * grad_weighted_filtered_patches
        grad_filter_weights = (grad_weighted_filtered_patches * filtered_patches).sum(0)
        #print("grad_filtered_patches", grad_filtered_patches)
        #print("grad_filter_weights", grad_filter_weights)
        grad_filter_bias = grad_filtered_patches.sum(0)
        grad_reshaped_patches = grad_filtered_patches
        #print("grad_filter_bias", grad_filter_bias)


        grad_patches = grad_reshaped_patches.view(
            batch_size, num_patches, channels, patch_size[0], patch_size[1]
        )
        #print("grad_patches", grad_patches)

        grad_input += patches_to_tensor2d(grad_patches, (height, width), patch_stride)
        #print("grad_input", grad_input)
        return grad_input, grad_filter_weights, grad_filter_bias, grad_compressor, None, None, None, None


if __name__ == "__main__":
    batch_size, channels, height, width = 4, 3, 32, 32
    patch_size = (8, 8)
    out_dim = 5
    compressor_size = (3, 3)
    padding = (1, 1)
    stride = (8, 8)
    patch_stride = (2, 2)
    tensor = torch.randn((batch_size, channels, height, width), requires_grad=True).to(float)
    batch_sample = tensor[0].unsqueeze(0)
    num_patches = ((height - patch_size[0]) // patch_stride[0] + 1) * ((width - patch_size[1]) // patch_stride[1] + 1)
    patch_weight = torch.randn((channels, patch_size[0], patch_size[1]), requires_grad=True).to(float)
    patch_bias = torch.randn((channels, patch_size[0], patch_size[1]), requires_grad=True).to(float)
    compressor = torch.randn((out_dim, compressor_size[0], compressor_size[1]), requires_grad=True).to(float)

    self_conv2d = SelfConv2d.apply(tensor, patch_weight, patch_bias, compressor, patch_size, patch_stride, padding, stride)
    self_conv2d_sample = SelfConv2d.apply(batch_sample, patch_weight, patch_bias, compressor, patch_size, patch_stride, padding, stride)
    print(f"Not mixing samples: {(self_conv2d_sample[0] == self_conv2d[0]).all()}")
    loss = self_conv2d.sum().backward()
    test = torch.autograd.gradcheck(SelfConv2d.apply, (tensor, patch_weight, patch_bias, compressor, patch_size, patch_stride, padding, stride))
    print(f"gradcheck passed: {test}")
