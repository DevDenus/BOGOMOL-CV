from torch import Tensor

def accuracy(predictions: Tensor, ground_truth: Tensor) -> float:
    predicted_classes = predictions.argmax(dim=1)
    correct = (predicted_classes == ground_truth).sum().item()
    accuracy = correct / ground_truth.size(0)
    return accuracy
