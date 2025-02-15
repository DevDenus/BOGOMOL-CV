import torch
from torch import Tensor
from torch.autograd import Function
from brevitas.function.ops import binary_sign

class BinarySignEdeFn(Function):
    """
    Binary sign function with gradient approximation according to IR-Net EDE
    """
    @staticmethod
    def forward(ctx, input : Tensor, k : Tensor, t : Tensor) -> Tensor:
        k, t = k.to(input.device), t.to(input.device)
        ctx.save_for_backward(input, k, t)
        out = binary_sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output) -> Tensor:
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None
