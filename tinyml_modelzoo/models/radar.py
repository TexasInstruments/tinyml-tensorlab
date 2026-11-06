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

"""Radar Applications models."""

import torch
import torch.nn as nn

# from ..utils import py_utils
# from .base import GenericModelWithSpec

class RF_MLP_3L_11K(nn.Module):
    """
    3-layer fully connected model for radar applications.

    Expected Input : (N, input_features)

    Architecture: BatchNorm -> Linear -> ReLU -> BatchNorm -> Linear -> ReLU ->
                  BatchNorm -> Linear
    Output: [batch, num_classes]
    """
    def __init__(self, config, input_features=130, variables=1, num_classes=2):
        super().__init__()

        self.config = config or {}
        self.input_features = self.config.get('input_features', input_features)
        self.variables = self.config.get('variables', variables)
        self.num_classes = self.config.get('num_classes', num_classes)

        input_size = self.input_features * self.variables

        self.bn1 = torch.nn.BatchNorm1d(num_features=input_size)
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=64)
        self.bn2 = torch.nn.BatchNorm1d(num_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=32)
        self.bn3 = torch.nn.BatchNorm1d(num_features=32)
        self.fc3 = torch.nn.Linear(in_features=32, out_features=self.num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.fc3(x)
        return x
    
class RF_MLP_4L_19K(nn.Module):
    """
    4-layer fully connected model for radar applications.

    Expected Input : (N, input_features)

    Architecture: BatchNorm -> Linear -> ReLU -> BatchNorm -> Linear -> ReLU ->
                 BatchNorm -> Linear -> ReLU -> BatchNorm -> Linear
    Output: [batch, num_classes]
    """
    def __init__(self, config, input_features=130, variables=1, num_classes=2):
        super().__init__()

        self.config = config or {}
        self.input_features = self.config.get('input_features', input_features)
        self.variables = self.config.get('variables', variables)
        self.num_classes = self.config.get('num_classes', num_classes)

        # Calculate input size after flattening
        input_size = self.input_features * self.variables

        # Build layers
        self.bn1 = torch.nn.BatchNorm1d(num_features=input_size)
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=64)
        self.bn2 = torch.nn.BatchNorm1d(num_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=32)
        self.bn3 = torch.nn.BatchNorm1d(num_features=32)
        self.fc3 = torch.nn.Linear(in_features=32, out_features=16)
        self.bn4 = torch.nn.BatchNorm1d(num_features=16)
        self.fc4 = torch.nn.Linear(in_features=16, out_features=self.num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.fc4(x)
        return x


class RF_DualStream_PointCloud_19K(nn.Module):
    """
    Dual-stream architecture that processes position and velocity features separately.
    Mainly for Point Cloud Applications with Tracks

    Architecture:
    - Position stream: Handles posz, acceleration, and point cloud data
    - Velocity stream: Handles velx, vely, velz
    - Fusion layers: Combines both streams for final classification

    Input: (N, input_features) [batch, frame_size * number of features in a radar frame]

    Output: [batch, num_classes]
    """

    def __init__(self, config, input_features=248, variables=1, num_classes=5):
        super().__init__()
        self.config = config or {}
        self.input_features = self.config.get('input_features', input_features)
        self.variables = self.config.get('variables', variables)
        self.num_classes = self.config.get('num_classes', num_classes)

        raw_frame_size = self.config.get('frame_size', None)

        self.passthrough_mode = raw_frame_size is None or raw_frame_size in (0, 'None', '')
        self.frame_size = 1 if self.passthrough_mode else raw_frame_size

        if not self.passthrough_mode:
            if self.input_features % self.frame_size != 0:
                raise ValueError(
                    f"input_features ({self.input_features}) must be divisible by frame_size ({self.frame_size})"
                )
            self.num_features_per_frame = self.input_features // self.frame_size

            input_size = self.input_features * self.variables
            output_size = self.num_classes
            num_velocity_features = 3  # track_velocity_x, track_velocity_y, track_velocity_z
            velocity_features_size = num_velocity_features * self.frame_size
            non_velocity_features_size = input_size - velocity_features_size


            # Position stream (processes non-velocity features)
            self.pos_stream = torch.nn.Sequential(
                torch.nn.BatchNorm1d(num_features=non_velocity_features_size),
                torch.nn.Linear(in_features=non_velocity_features_size, out_features=60),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=60)
            )

            # Velocity stream (processes velocity features)
            self.vel_stream = torch.nn.Sequential(
                torch.nn.BatchNorm1d(num_features=velocity_features_size),
                torch.nn.Linear(in_features=velocity_features_size, out_features=20),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=20)
            )

            # Fusion layers (combines 60 + 20 = 80 features)
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(in_features=80, out_features=40),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=40),
                torch.nn.Linear(in_features=40, out_features=20),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=20),
                torch.nn.Linear(in_features=20, out_features=output_size)
            )
        else:
            # Passthrough mode: input is already per-frame features
            # We'll use a simpler MLP that treats all features equally
            input_size = self.input_features * self.variables
            self.mlp = torch.nn.Sequential(
                torch.nn.BatchNorm1d(num_features=input_size),
                torch.nn.Linear(in_features=input_size, out_features=64),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=64),
                torch.nn.Linear(in_features=64, out_features=32),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=32),
                torch.nn.Linear(in_features=32, out_features=self.num_classes)
            )

    def forward(self, x):
        if self.passthrough_mode:
            # Passthrough mode: x is [batch, features_per_frame]
            out = self.mlp(x)
        else:
            # Windowed mode: x is [batch, frame_size * features_per_frame]
            pos_features = []
            vel_features = []

            # Process each frame
            for i in range(self.frame_size):
                frame_start = i * self.num_features_per_frame

                pos_idx = [frame_start] + list(range(frame_start + 4, frame_start + self.num_features_per_frame))
                pos_features.append(x[:, pos_idx])

                vel_idx = list(range(frame_start + 1, frame_start + 4))
                vel_features.append(x[:, vel_idx])

            pos = torch.cat(pos_features, dim=1) 
            vel = torch.cat(vel_features, dim=1)  

            # Process through separate streams
            pos_out = self.pos_stream(pos)  
            vel_out = self.vel_stream(vel) 

            # Fuse and classify
            combined = torch.cat([pos_out, vel_out], dim=1)  # [batch, 80]
            out = self.fusion(combined)  # [batch, output_size]
        
        return out


# Export all radar models
__all__ = [
    'RF_MLP_3L_11K', 'RF_MLP_4L_19K', 'RF_DualStream_PointCloud_19K'
]