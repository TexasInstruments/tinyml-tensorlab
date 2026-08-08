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

"""Radar Point Cloud models."""

import torch

from ..utils import py_utils
from .base import GenericModelWithSpec

class LINEAR_4L_PC(GenericModelWithSpec): #can come back to naming convention later
    def __init__(self, config, input_features= 176, variables=1, num_classes=5):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        input_size = self.input_features * self.variables #(just int for now)
        self.bn1 = torch.nn.BatchNorm1d(num_features= input_size)
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=64)
        self.bn2 = torch.nn.BatchNorm1d(num_features= 64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=32)
        self.bn3 = torch.nn.BatchNorm1d(num_features=32)
        self.fc3 = torch.nn.Linear(in_features=32, out_features=16)
        self.bn4 = torch.nn.BatchNorm1d(num_features=16)
        self.fc4 = torch.nn.Linear(in_features=16, out_features=self.num_classes)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1) #flatten
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.fc3(x)
        x = self.bn4(x)
        x = self.fc4(x)
        x = torch.softmax(x, dim=1)
        return x


# Export all classification models
__all__ = [
    'LINEAR_4L_PC',
]