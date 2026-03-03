"""ResNet models for CIFAR-10."""

import torch.nn as nn
import torchvision


def get_resnet18_cifar10(pretrained: bool = False) -> nn.Module:
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images).

    Modifications from standard ResNet-18:
    - Replace first 7x7 conv with 3x3 conv (stride=1, no max pooling)
    - Remove first max pooling layer
    - Change output layer to 10 classes

    Args:
        pretrained: Whether to load pretrained weights (not supported for CIFAR-10 variant)

    Returns:
        Modified ResNet-18 model
    """
    # Load standard ResNet-18 architecture
    model = torchvision.models.resnet18(weights=None)

    # Adapt for CIFAR-10 (32x32 images)
    # Replace 7x7 conv with 3x3 conv (stride=1, no downsampling)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    # Remove max pooling (it's too aggressive for 32x32 images)
    model.maxpool = nn.Identity()

    # Change output layer to 10 classes (CIFAR-10)
    model.fc = nn.Linear(512, 10)

    return model
