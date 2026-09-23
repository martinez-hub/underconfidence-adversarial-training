"""ResNet models for CIFAR-10."""

from typing import Sequence

import torch
import torch.nn as nn
import torchvision

# CIFAR-10 channel statistics. These live here rather than in the data pipeline
# because normalization is applied *inside* the model (see NormalizedModel), so
# that adversarial attacks operate on raw [0, 1] pixels.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class InputNormalize(nn.Module):
    """
    Per-channel input normalization as a model layer.

    Keeping normalization inside the model means every external consumer --
    most importantly the adversarial attacks -- sees images in raw [0, 1] pixel
    space. That is what makes an L-infinity budget of epsilon=8/255 and the
    ``clamp(x, 0, 1)`` projection in the attacks mean what they claim to mean.
    Normalizing in the data pipeline instead would leave the attacks operating
    on tensors spanning roughly [-2.43, 2.75], where neither the clamp nor the
    epsilon-ball is the constraint the paper describes.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        # persistent=False: these are fixed dataset constants, not learned
        # state. Keeping them out of state_dict means checkpoints hold only
        # trained weights, and checkpoints saved before normalization moved
        # into the model still load.
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class NormalizedModel(nn.Module):
    """
    Wraps a backbone so that it accepts raw [0, 1] images.

    Attributes:
        normalize: The input normalization layer.
        backbone: The wrapped classifier.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mean: Sequence[float] = CIFAR10_MEAN,
        std: Sequence[float] = CIFAR10_STD,
    ):
        super().__init__()
        self.normalize = InputNormalize(mean, std)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(x))


def get_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images), accepting [0, 1] inputs.

    Modifications from standard ResNet-18:
    - Replace first 7x7 conv with 3x3 conv (stride=1, no downsampling)
    - Remove first max pooling layer
    - Change output layer to ``num_classes`` classes
    - Normalize inputs inside the model, so ``model(x)`` expects x in [0, 1]

    Args:
        num_classes: Number of output classes

    Returns:
        Modified ResNet-18 wrapped in a NormalizedModel

    Raises:
        ValueError: If num_classes is not positive
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    # Load standard ResNet-18 architecture
    backbone = torchvision.models.resnet18(weights=None)

    # Adapt for CIFAR-10 (32x32 images)
    # Replace 7x7 conv with 3x3 conv (stride=1, no downsampling)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove max pooling (it's too aggressive for 32x32 images)
    backbone.maxpool = nn.Identity()

    # Change output layer to num_classes classes
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

    return NormalizedModel(backbone)
