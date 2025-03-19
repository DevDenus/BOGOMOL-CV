import torch
from torch import Tensor
from torch.autograd import Function

class ReLU2SignEde(Function):
    @staticmethod
    def forward(ctx, input : Tensor, a : Tensor) -> Tensor:
        a = a.to(input.device)
        ctx.save_for_backward(input, a)
        out = (1. - a)*torch.ones_like(input) + a*input
        out[input < 0] = a - 1.
        return out

    @staticmethod
    def backward(ctx, grad_output : Tensor) -> Tensor:
        input, a = ctx.saved_tensors
        grad_input = torch.zeros_like(input, device=input.device)
        grad_input[input >= 0] = a
        grad_input = grad_input * grad_output
        return grad_input, None

class Linear2SignEde(Function):
    @staticmethod
    def forward(ctx, input : Tensor, a : Tensor) -> Tensor:
        a = a.to(input.device).to(input.dtype)
        ctx.save_for_backward(input, a)
        out = a*torch.ones_like(input)
        out[input > 1/a] = 1.
        out[input < -1/a] = -1.
        return out

    @staticmethod
    def backward(ctx, grad_output : Tensor) -> Tensor:
        input, a = ctx.saved_tensors
        grad_input = torch.zeros_like(input, device=input.device)
        grad_input[abs(input) <= 1/a] = a
        grad_input = grad_input * grad_output
        return grad_input, None

class Tanh2SignEde(Function):
    @staticmethod
    def forward(ctx, input : Tensor, k : Tensor, t : Tensor) -> Tensor:
        k = k.to(input.device)
        t = t.to(input.device)
        ctx.save_for_backward(input, k, t)
        out = k * torch.tanh(input * t)
        return out

    @staticmethod
    def backward(ctx, grad_output : Tensor) -> Tensor:
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None
