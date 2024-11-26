from typing import Any
import torch

from torch.autograd import Function

class BinaryQuantize(Function):
    """
    EDE from IR-Net
    """
    @staticmethod
    def forward(ctx, input, k, t) -> Any:
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output) -> Any:
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

class ZhegalkinTransform1d(Function):
    """
    Transforming binary vector of shape (1,n) to shape (1, n*(n-1)/2), containing elements x_i*x_j
    """
    @staticmethod
    def forward(ctx, input) -> Any:
        ctx.save_for_backward(input)
        result = []
        for i in range(input.shape[1]):
            result.append(input[:,i])
            for j in range(i+1, input.shape[1]):
                result.append(torch.bitwise_and(input[:,i], input[:,j]))
        return torch.tensor(result).bool()

    @staticmethod
    def backward(ctx, grad_outputs) -> Any:
        input = ctx.saved_tensors
        grad_inputs = torch.zeros_like(input)
        for i in range(input.shape[1]):
            mask = torch.zeros_like(grad_outputs)
            mask[:,i] += torch.ones_like(mask[:,i])
            for j in range(input.shape[1]):
                if j == i:
                    continue
                mask[:,j] += input[:,j]
            grad_inputs += mask * grad_outputs.transpose()
        return grad_inputs

class ZhegalkinTransform2d(Function):
    """
    Transforming binary array of shape (m,n) to
    """
    @staticmethod
    def forward(ctx, input, filter_size : int = 3, padding : int = 0, stride : int = 1) -> Any:
        assert input.shape[0] >= filter_size and input.shape[1] >= filter_size
        ctx.save_for_backward(input, filter_size, padding, stride)

    @staticmethod
    def backward(ctx, grad_outputs) -> Any:
        input, filter_size, padding, stride = ctx.saved_tensors

class BMatrixMul(Function):
    """
    Analogue of matrix multiplication with XOR and AND as summation and multiplication.
    """
    @staticmethod
    def forward(ctx, input, weights, bias = None):
        ctx.save_for_backward(input, weights, bias)
        x_bin = (input > 0).bool()
        weights_bin = (weights > 0).bool()

        and_result = torch.bitwise_and(x_bin.unsqueeze(1), weights_bin.unsqueeze(0).bool()).float()
        result = and_result.sum(dim=-1) % 2
        if bias:
            result = (result + bias) % 2

        return result.float()

    @staticmethod
    def backward(ctx, grad_outputs):
        input, weights, bias = ctx.saved_tensors
        grad_inputs = torch.zeros_like(input)
        return grad_inputs

class BConv2d(Function):
    @staticmethod
    def forward(ctx, input):
        pass

    @staticmethod
    def backward(ctx, grad_outputs):
        pass
