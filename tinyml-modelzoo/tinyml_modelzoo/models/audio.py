"""Audio classification models."""

import torch

from .base import GenericModelWithSpec


class CNN_AUDIO_DSCNN(GenericModelWithSpec):
    """
    DSCNN model for Google Speech Commands.

    Expected input:
        (N, 1, 49, 10)

    Architecture:
        Conv10x4/s2 -> Dropout ->
        [Depthwise3x3 + Pointwise1x1] x 4 ->
        Dropout -> AdaptiveAvgPool -> FC
    """

    def __init__(self, config, input_features=(49, 10), variables=1, num_classes=12):
        super().__init__(
            config,
            input_features=input_features,
            variables=variables,
            num_classes=num_classes,
        )

        filters = int(getattr(config, "filters", 64)) if config is not None else 64

        spectrogram_length = self.input_features[0]
        dct_coefficient_count = self.input_features[1]

        pads = (
            4 + spectrogram_length % 2,
            1 + dct_coefficient_count % 2,
        )

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.variables,
                out_channels=filters,
                kernel_size=(10, 4),
                stride=(2, 2),
                padding=pads,
            ),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),

            torch.nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),

            torch.nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),

            torch.nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),

            torch.nn.Dropout(0.4),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(filters, self.num_classes),
        )

    def forward(self, x):
        return self.layers(x)

# Export all classification models
__all__ = [
    'CNN_AUDIO_DSCNN',
]

