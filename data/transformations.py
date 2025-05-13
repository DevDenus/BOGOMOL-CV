from typing import Tuple

from torchvision.transforms import v2

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(input_size : Tuple[int], mean : Tuple[float], std = Tuple[float]):
    transform_train = v2.Compose([
        v2.Resize(input_size),
        v2.RandomCrop(input_size, padding=4),
        v2.RandomHorizontalFlip(),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.ToTensor(),
        v2.Normalize(
            mean=mean,
            std=std
        )
    ])
    return transform_train

def val_transform(input_size : Tuple[int], mean : Tuple[float], std = Tuple[float]):
    transform_val = v2.Compose([
        v2.Resize(input_size),
        v2.ToTensor(),
        v2.Normalize(
            mean=mean,
            std=std
        )
    ])
    return transform_val

def cutmix_or_mixup(num_classes : int, alpha_cutmix : float = 1., alpha_mixup : float = 0.2):
    cutmix = v2.CutMix(alpha=alpha_cutmix, num_classes=num_classes)
    mixup = v2.MixUp(alpha=alpha_mixup, num_classes=num_classes)
    return v2.RandomChoice([cutmix, mixup])
