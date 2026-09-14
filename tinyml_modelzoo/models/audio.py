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

class CNN_AUDIO_DSCNN_32K_NPU(GenericModelWithSpec):
    """
    SRAM/Flash-optimized DSCNN for LPC audio features on MSPM0G5187.

    Designed for (N, 1, 100, 70) LPC input (8820 Hz, 2s window).

    Target:
        SRAM < 32 KB
        Flash < 128 KB
        W8 quantization

    Architecture:
        FCONV 10x4 / s(4,4)       <- stride-4 stem keeps largest buffer to ~22 KB at 48ch
        -> DW 3x3 / s(2,2) + PW   <- further halves to ~6 KB
        -> [DW 3x3 + PW] x 3
        -> Global AvgPool
        -> FC

    Channels: 48 throughout
    Peak SRAM: 48*(26*18) + 48*(13*9) = 22464 + 5616 = 28080 bytes < 32 KB
    """

    def __init__(
        self,
        config,
        input_features=(100, 70),
        variables=1,
        num_classes=12,
    ):
        super().__init__(
            config,
            input_features=input_features,
            variables=variables,
            num_classes=num_classes,
        )

        filters = 48

        self.layers = torch.nn.Sequential(

            # Stem: stride-4 in both dims to keep output buffer small
            # (100,70) -> (26,18) at 48ch = 22,464 bytes
            torch.nn.Conv2d(
                in_channels=self.variables,
                out_channels=filters,
                kernel_size=(10, 4),
                stride=(4, 4),
                padding=(5, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            # DS block 1: stride (2,2) -> (13,9) at 48ch = 5,616 bytes
            torch.nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=filters, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),

            # DS block 2 (stride 1, stays at 13x9)
            torch.nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),

            # DS block 3
            torch.nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),

            # DS block 4
            torch.nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),

            torch.nn.Dropout(0.4),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(filters, self.num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class TCDSBasicBlock(torch.nn.Module):
    """Temporal Channel-Decoupled Separable block (no residual — avoids quantized-add issues)."""

    def __init__(self, in_fmsize=16, out_fmsize=16, kernel=9, stride=2):
        super().__init__()
        pad = (kernel - 1) // 2

        self.conv_1_dw = torch.nn.Conv2d(in_fmsize, in_fmsize, (kernel, 1), stride=(stride, 1), padding=(pad, 0), groups=in_fmsize)
        self.bn_1_dw = torch.nn.BatchNorm2d(in_fmsize)
        self.act_1_dw = torch.nn.ReLU()
        self.conv_1_pw = torch.nn.Conv2d(in_fmsize, out_fmsize, 1)
        self.bn_1_pw = torch.nn.BatchNorm2d(out_fmsize)
        self.act_1_pw = torch.nn.ReLU()

        self.conv_2_dw = torch.nn.Conv2d(out_fmsize, out_fmsize, (kernel, 1), padding=(pad, 0), groups=out_fmsize)
        self.bn_2_dw = torch.nn.BatchNorm2d(out_fmsize)
        self.act_2_dw = torch.nn.ReLU()
        self.conv_2_pw = torch.nn.Conv2d(out_fmsize, out_fmsize, 1)
        self.bn_2_pw = torch.nn.BatchNorm2d(out_fmsize)
        self.act_2_pw = torch.nn.ReLU()

    def forward(self, x):
        x = self.act_1_dw(self.bn_1_dw(self.conv_1_dw(x)))
        x = self.act_1_pw(self.bn_1_pw(self.conv_1_pw(x)))
        x = self.act_2_dw(self.bn_2_dw(self.conv_2_dw(x)))
        x = self.act_2_pw(self.bn_2_pw(self.conv_2_pw(x)))
        return x


class CNN_AUDIO_TCDS_ResNet_NPU(GenericModelWithSpec):
    """
    Temporal Channel-Decoupled Separable ResNet for LPC audio features.

    SRAM-efficient design: permutes input (N,1,T,F) → (N,F,T,1) so that
    LPC coefficients become channels and all convolutions are 1D over time.
    No residual connections — avoids quantized-add incompatibility with TI QAT.
    Intermediate buffers are (channels × T × 1) — never 2D — keeping peak
    SRAM in the low single-digit KB range.

    Expected input: (N, 1, 100, 70)  [100 time frames × 70 LPC features]
    Default: fmsize=16, channels grow 16→24→32→48, 3 stride-2 blocks.
    Peak SRAM: ~3–4 KB  (vs 83+ KB for 2D-spatial DSCNN models)
    """

    def __init__(self, config, input_features=(100, 70), variables=1, num_classes=12):
        super().__init__(
            config,
            input_features=input_features,
            variables=variables,
            num_classes=num_classes,
        )

        fmsize = int(getattr(config, 'fmsize', 16)) if config is not None else 16
        kernel = 9

        # LPC coefficients become the channel dimension after permute
        num_lpc = self.input_features[1] if self.input_features is not None else 70

        # Input BN on raw (N,1,T,F) — must be BEFORE the permute so it appears
        # as the first ONNX node. The TI NPU compiler scans from the graph input
        # to find a normalization sequence; Transpose before BN breaks detection.
        self.bn_input = torch.nn.BatchNorm2d(1)

        # Initial 1D temporal conv: (N, num_lpc, T, 1) → (N, fmsize, T, 1)
        self.conv1 = torch.nn.Conv2d(num_lpc, fmsize, (3, 1), padding=(1, 0), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(fmsize)
        self.act1 = torch.nn.ReLU()

        # Three stride-2 + three stride-1 residual blocks; channels grow 16→24→32→48
        layer_fmsize = [fmsize, int(fmsize * 1.5), int(fmsize * 2.0), int(fmsize * 3.0)]
        self.blocks = torch.nn.Sequential(
            TCDSBasicBlock(layer_fmsize[0], layer_fmsize[1], kernel=kernel, stride=2),
            TCDSBasicBlock(layer_fmsize[1], layer_fmsize[1], kernel=kernel, stride=1),
            TCDSBasicBlock(layer_fmsize[1], layer_fmsize[2], kernel=kernel, stride=2),
            TCDSBasicBlock(layer_fmsize[2], layer_fmsize[2], kernel=kernel, stride=1),
            TCDSBasicBlock(layer_fmsize[2], layer_fmsize[3], kernel=kernel, stride=2),
            TCDSBasicBlock(layer_fmsize[3], layer_fmsize[3], kernel=kernel, stride=1),
        )

        # Global average pool over the time axis → (N, channels, 1, 1), then flatten to channels
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc = torch.nn.Linear(layer_fmsize[3], self.num_classes)

    def forward(self, x):
        # BN first on raw (N,1,T,F) — keeps it as first ONNX node for TI compiler
        x = self.bn_input(x)
        # Permute so LPC coefficients become channels; input is contiguous after BN
        x = x.permute(0, 3, 2, 1).contiguous()   # (N, 1, T, F) → (N, F, T, 1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.pool(x)             # (N, channels, 1, 1)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


# Export all classification models
__all__ = [
    'CNN_AUDIO_DSCNN',
    'CNN_AUDIO_DSCNN_32K_NPU',
    'CNN_AUDIO_TCDS_ResNet_NPU',
]

