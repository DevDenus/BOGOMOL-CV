from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Function

class EDEstimator(Function):
    @staticmethod
    def forward(*args, **kwargs):
        return 
