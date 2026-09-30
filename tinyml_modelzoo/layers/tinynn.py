#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################

# only class types imported or defined here and part of __all__ can be part of model_spec
# define custom modules here if needed and add the names to __all__


import copy

import numpy as np
import torch.nn


__all__ = ['IdentityLayer', 'ConvLayer', 'BatchNormLayer', 'ReLULayer', 'MaxPoolLayer', 'AvgPoolLayer',
           'AdaptiveAvgPoolLayer', 'LinearLayer', 'LinearBNReLULayer', 'ConvBNReLULayer', 'ReshapeLayer',
           'RNNLayer', 'GRULayer', 'LSTMLayer',]


def compute_output_size(type, input_tensor_size, kernel_size, padding, stride, dilation, **kwargs):
    output_tensor_size = copy.deepcopy(input_tensor_size)
    if type in ('ConvLayer','ConvBNReLULayer'):
        out_channels = kwargs['out_channels']
        output_tensor_size[1] = out_channels
    #
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    output_tensor_size[2] = int(((input_tensor_size[2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
    output_tensor_size[3] = int(((input_tensor_size[3] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
    return output_tensor_size


def IdentityLayer(input_tensor_size=None, **kwargs):
    layer = torch.nn.Identity(**kwargs)
    output_tensor_size = copy.deepcopy(input_tensor_size)
    return layer, output_tensor_size


def BatchNormLayer(input_tensor_size=None, **kwargs):
    layer = torch.nn.BatchNorm2d(**kwargs)
    output_tensor_size = copy.deepcopy(input_tensor_size)
    return layer, output_tensor_size


def ReLULayer(input_tensor_size=None, **kwargs):
    layer = torch.nn.ReLU(**kwargs)
    output_tensor_size = copy.deepcopy(input_tensor_size)
    return layer, output_tensor_size


def ConvLayer(kernel_size, padding=None, stride=1, input_tensor_size=None, dilation=1, **kwargs):
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2, 'kernel_size must be a tuple of size 2'
        if padding is None:
            padding = tuple([k//2 for k in kernel_size])
        elif isinstance(padding, int):
            padding = (padding, 0)
        #
        if isinstance(stride, int):
            stride = (stride, 1)
        #
        if isinstance(dilation, int):
            dilation = (dilation, 1)
        #
        layer = torch.nn.Conv2d(kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, **kwargs)
        output_tensor_size = compute_output_size(type='ConvLayer', input_tensor_size=input_tensor_size,
                                                 kernel_size=kernel_size, padding=padding, stride=stride,
                                                 dilation=dilation, **kwargs)
        return layer, output_tensor_size


def MaxPoolLayer(kernel_size, padding=None, stride=1, input_tensor_size=None, dilation=1, **kwargs):
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2, 'kernel_size must be a tuple of size 2'
        if padding is None:
            padding = tuple([k//2 for k in kernel_size])
        elif isinstance(padding, int):
            padding = (padding, 0)
        #
        if isinstance(stride, int):
            stride = (stride, 1)
        #
        if isinstance(dilation, int):
            dilation = (dilation, 1)
        #
        layer = torch.nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, **kwargs)
        output_tensor_size = compute_output_size(type='MaxPoolLayer', input_tensor_size=input_tensor_size,
                                                 kernel_size=kernel_size, dilation=dilation,
                                                 padding=padding, stride=stride, **kwargs)
        return layer, output_tensor_size


def AvgPoolLayer(kernel_size, padding=None, stride=1, input_tensor_size=None, dilation=1, **kwargs):
    assert isinstance(kernel_size, tuple) and len(kernel_size) == 2, 'kernel_size must be a tuple of size 2'
    if padding is None:
        padding = tuple([k // 2 for k in kernel_size])
    elif isinstance(padding, int):
        padding = (padding, 0)
    #
    if isinstance(stride, int):
        stride = (stride, 1)
    #
    if isinstance(dilation, int):
        dilation = (dilation, 1)
    #
    layer = torch.nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride, **kwargs)
    output_tensor_size = compute_output_size(type='AvgPoolLayer', input_tensor_size=input_tensor_size,
                                             kernel_size=kernel_size, dilation=dilation,
                                             padding=padding, stride=stride, **kwargs)
    return layer, output_tensor_size


def AdaptiveAvgPoolLayer(output_size, input_tensor_size=None, **kwargs):
    assert isinstance(output_size, tuple) and len(output_size) == 2, 'output_size must be a tuple of size 2'
    layer = torch.nn.AdaptiveAvgPool2d(output_size=output_size, **kwargs)
    output_tensor_size = copy.deepcopy(input_tensor_size)
    output_tensor_size[-2] = output_size[-2]
    output_tensor_size[-1] = output_size[-1]
    return layer, output_tensor_size


def ConvBNReLULayer(in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, input_tensor_size=None, **kwargs):
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2, 'kernel_size must be a tuple of size 2'
        if padding is None:
            padding = tuple([k//2 for k in kernel_size])
        elif isinstance(padding, int):
            padding = (padding, 0)
        #
        if isinstance(stride, int):
            stride = (stride, 1)
        #
        if isinstance(dilation, int):
            dilation = (dilation, 1)
        #
        layers = []
        layers += [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, dilation=dilation, **kwargs)]
        layers += [torch.nn.BatchNorm2d(num_features=out_channels)]
        layers += [torch.nn.ReLU()]
        layer = torch.nn.Sequential(*layers)
        output_tensor_size = compute_output_size(type='ConvBNReLULayer', input_tensor_size=input_tensor_size,
                                                 out_channels=out_channels, kernel_size=kernel_size,
                                                 padding=padding, stride=stride, dilation=dilation, **kwargs)
        return layer, output_tensor_size


class _ReshapeLayer(torch.nn.Module):
    def __init__(self, ndim=2):
        super().__init__()
        self.ndim = ndim

    def forward(self, x):
        if self.ndim == 2:
            return x.reshape(x.shape[0], -1)
        else:
            assert False, f'Unsupported value of ndim {self.ndim}'


def ReshapeLayer(ndim=2, input_tensor_size=None):
    layer = _ReshapeLayer(ndim=ndim)
    if ndim == 2:
        output_tensor_size = [input_tensor_size[0], np.prod(input_tensor_size[1:])]
    else:
        assert ndim==2, f'ndim must be 2 for now. got {ndim}'
    return layer, output_tensor_size


def LinearLayer(input_tensor_size=None, out_features=None, **kwargs):
    layer = torch.nn.Linear(out_features=out_features, **kwargs)
    output_tensor_size = copy.deepcopy(input_tensor_size)
    output_tensor_size[-1] = out_features
    return layer, output_tensor_size


def LinearBNReLULayer(in_features, out_features, input_tensor_size=None, **kwargs):
    """Linear + BatchNorm1d + ReLU combined layer."""
    layers = []
    layers += [torch.nn.Linear(in_features=in_features, out_features=out_features, **kwargs)]
    layers += [torch.nn.BatchNorm1d(num_features=out_features)]
    layers += [torch.nn.ReLU()]
    layer = torch.nn.Sequential(*layers)
    output_tensor_size = copy.deepcopy(input_tensor_size)
    output_tensor_size[-1] = out_features
    return layer, output_tensor_size


def ReluLayer(input_tensor_size=None):
    layer = torch.nn.ReLU()
    output_tensor_size = copy.deepcopy(input_tensor_size)
    return layer, output_tensor_size

def DropoutLayer(dropout_prob, input_tensor_size=None):
    layer = torch.nn.Dropout(dropout_prob)
    output_tensor_size = copy.deepcopy(input_tensor_size)
    return layer, output_tensor_size


class TCDSBasicBlock(torch.nn.Module):
    """
    Depthwise-separable residual block with 4-conv main branch and 1-conv projection skip.

    Main branch: DW(k) -> BN -> ReLU -> PW(1) -> BN -> ReLU -> DW(k) -> BN -> ReLU -> PW(1) -> BN
    Skip branch: Conv1x1 -> BN  (projection when stride != 1 or channels change), else Identity
    Output: ReLU(main + skip)
    """
    def __init__(self, in_channels, out_channels, kernel_t, stride_t):
        super().__init__()
        pad = (kernel_t - 1) // 2

        if stride_t != 1 or in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride=(stride_t, 1)),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            )
        else:
            self.skip = torch.nn.Identity()

        self.dw1 = torch.nn.Conv2d(in_channels, in_channels, (kernel_t, 1),
                                   stride=(stride_t, 1), padding=(pad, 0), groups=in_channels)
        self.bn_dw1 = torch.nn.BatchNorm2d(in_channels)
        self.act_dw1 = torch.nn.ReLU()
        self.pw1 = torch.nn.Conv2d(in_channels, out_channels, 1)
        self.bn_pw1 = torch.nn.BatchNorm2d(out_channels)
        self.act_pw1 = torch.nn.ReLU()
        self.dw2 = torch.nn.Conv2d(out_channels, out_channels, (kernel_t, 1),
                                   padding=(pad, 0), groups=out_channels)
        self.bn_dw2 = torch.nn.BatchNorm2d(out_channels)
        self.act_dw2 = torch.nn.ReLU()
        self.pw2 = torch.nn.Conv2d(out_channels, out_channels, 1)
        self.bn_pw2 = torch.nn.BatchNorm2d(out_channels)
        self.act_out = torch.nn.ReLU()

    def forward(self, x):
        skip = self.skip(x)
        x = self.act_dw1(self.bn_dw1(self.dw1(x)))
        x = self.act_pw1(self.bn_pw1(self.pw1(x)))
        x = self.act_dw2(self.bn_dw2(self.dw2(x)))
        x = self.bn_pw2(self.pw2(x))
        return self.act_out(x + skip)


def TCDSBasicBlockLayer(in_channels, out_channels, kernel_size, stride=2, input_tensor_size=None, **kwargs):
    assert isinstance(kernel_size, tuple) and len(kernel_size) == 2, 'kernel_size must be a tuple of size 2'
    if isinstance(stride, int):
        stride = (stride, 1)
    kernel_t, stride_t = kernel_size[0], stride[0]
    pad = (kernel_t - 1) // 2

    layer = TCDSBasicBlock(in_channels, out_channels, kernel_t, stride_t)
    output_tensor_size = copy.deepcopy(input_tensor_size)
    output_tensor_size[1] = out_channels
    output_tensor_size[2] = (input_tensor_size[2] + 2 * pad - kernel_t) // stride_t + 1
    return layer, output_tensor_size


def RNNLayer(input_size, hidden_size, return_last_timestep=False, input_tensor_size=None, num_layers=1, dropout=0.1, batch_first=True, rnn_type='RNN', **kwargs):

    if rnn_type.upper() == 'LSTM':
        rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif rnn_type.upper() == 'GRU':
        rnn = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:  # Default to simple RNN
        rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            num_layers=num_layers,
            dropout=dropout,
        )

    # Wrapping the RNN layer in a custom class to include return_last_timestep
    class RNNWithReturnLastTimestep(torch.nn.Module):
        def __init__(self, rnn, return_last_timestep):
            super().__init__()
            self.rnn = rnn
            self.return_last_timestep = return_last_timestep

        def forward(self, x, h_0=None, c_0=None):
            # Forward pass through the RNN layer

            x = x.transpose(1,2)  # Reshape to (batch, seq_len, features)
            x = x[:, :, :,0]

            x,_=self.rnn(x)

            # If return_last_timestep is True, return only the last timestep
            if self.return_last_timestep:
                x = x[:, -1, :]
            else:
                x=x.transpose(1,2) # Reshape back to (batch, features, seq_len)
                x = x[:, :, :, None] # Add back the last dimension
            return x

    # Wrap the RNN layer
    layer = RNNWithReturnLastTimestep(rnn, return_last_timestep)

    if return_last_timestep:
        output_tensor_size = [input_tensor_size[0], hidden_size]
    else:
        output_tensor_size = [input_tensor_size[0], hidden_size, input_tensor_size[2], input_tensor_size[3]]

    return layer, output_tensor_size

def LSTMLayer(input_size, hidden_size, return_last_timestep=False, input_tensor_size=None, num_layers=1, dropout=0.1, batch_first=True, **kwargs):
    return RNNLayer(input_size, hidden_size, return_last_timestep, input_tensor_size, num_layers, dropout, batch_first, rnn_type='LSTM', **kwargs)

def GRULayer(input_size, hidden_size, return_last_timestep=False, input_tensor_size=None, num_layers=1, dropout=0.1, batch_first=True, **kwargs):
    return RNNLayer(input_size, hidden_size, return_last_timestep, input_tensor_size, num_layers, dropout, batch_first, rnn_type='GRU', **kwargs)
