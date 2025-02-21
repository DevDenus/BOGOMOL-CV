import torch
from torch import Tensor
from torch.autograd import Function

class BinarySignEdeFn(Function):
    """
    Binary sign function with gradient approximation according to IR-Net EDE
    """
    @staticmethod
    def forward(ctx, input : Tensor, a : Tensor) -> Tensor:
        a = a.to(input.device)
        ctx.save_for_backward(input, a)
        out = (1-a)*torch.ones_like(input) + a*input
        out[input < 0] = a-1
        return out

    @staticmethod
    def backward(ctx, grad_output : Tensor) -> Tensor:
        input, a = ctx.saved_tensors
        grad_input = torch.zeros_like(input, device=input.device)
        grad_input[input >= 0] = a
        grad_input = grad_input * grad_output
        return grad_input, None
