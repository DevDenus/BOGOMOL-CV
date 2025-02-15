from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.core.bit_width import BitWidthConst
from brevitas.core.quant.delay import DelayWrapper
from brevitas.core.utils import StatelessBuffer

from src.quant.ops import BinarySignEdeFn

class EdeBinaryQuant(brevitas.jit.ScriptModule):
    def __init__(self, scaling_impl: Module, signed: bool = True, quant_delay_steps: int = 0):
        super().__init__()
        assert signed, "Unsigned binary quant not supported"
        self.scaling_impl = scaling_impl
        self.bit_width = BitWidthConst(1)
        self.zero_point = StatelessBuffer(torch.tensor(0.0))
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method
    def forward(self, x: Tensor, k : Tensor, t : Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(x)
        y = BinarySignEdeFn.apply(x, k, t) * scale
        y = self.delay_wrapper(x, y)
        return y, scale, self.zero_point(), self.bit_width()
