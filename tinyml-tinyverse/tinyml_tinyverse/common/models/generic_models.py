#################################################################################
# Copyright (c) 2023-2024, Texas Instruments
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

import yaml
from .generic_classification_models import *
from .generic_regression_models import *
from .generic_feature_extraction_models import *
from .generic_autoencoder_models import *


if __name__ == '__main__':
    yaml.Dumper.ignore_aliases = lambda *data: True
    filename = 'CNN_TS_GEN_BASE_13K.yaml'
    with open(filename, 'w') as fp:
        yaml.dump(CNN_TS_GEN_BASE_13K(dict()).model_spec, fp, default_flow_style=False, sort_keys=False)

    filename = 'CNN_TS_GEN_BASE_6K.yaml'
    with open(filename, 'w') as fp:
        yaml.dump(CNN_TS_GEN_BASE_6K(dict()).model_spec, fp, default_flow_style=False, sort_keys=False)

    filename = 'CNN_TS_GEN_BASE_4K.yaml'
    with open(filename, 'w') as fp:
        yaml.dump(CNN_TS_GEN_BASE_4K(dict()).model_spec, fp, default_flow_style=False, sort_keys=False)

    filename = 'CNN_TS_GEN_BASE_1K.yaml'
    with open(filename, 'w') as fp:
        yaml.dump(CNN_TS_GEN_BASE_1K(dict()).model_spec, fp, default_flow_style=False, sort_keys=False)
