from torch import nn, Tensor

crossentropy = nn.CrossEntropyLoss()

def soft_crossentropy(pred : Tensor, soft_targets : Tensor) -> Tensor:
    soft_targets = soft_targets.clamp(min=1e-7)
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
    log_probs = nn.functional.log_softmax(pred, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()
