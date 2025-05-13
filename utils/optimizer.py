import math
from typing import Tuple

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def configure_optimizer(
    model,
    base_lr : float = 3e-4,
    weight_decay : float = 0.1,
    betas : Tuple[float] = (0.9, 0.999),
    warmup_epochs : int = 5,
    total_epochs : int = 100,
    steps_per_epoch : int = 1000,
    num_cycles : int = 1
):
    # Weight decay for normalization and bias parameters are harmful
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif "alpha" in name.lower():
            no_decay_params.append(param)
        elif param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=base_lr, betas=betas)

    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    decay_steps = total_steps - warmup_steps
    cycle_steps = decay_steps // num_cycles

    # Cyclical cosine annealing with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        step_in_decay = step - warmup_steps
        cycle_position = step_in_decay % cycle_steps
        cosine_progress = cycle_position / cycle_steps
        return 0.5 * (1 + math.cos(math.pi * cosine_progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler
