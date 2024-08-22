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

import os
import traceback
from logging import getLogger

from ..utils import misc_utils

from .generic_models import *

# from .kilby_models import *

model_dict = {
    'CNN_TS_GEN_BASE_3K': CNN_TS_GEN_BASE_3K,
    'CNN_TS_GEN_BASE_7K': CNN_TS_GEN_BASE_7K,
    # 'CNN_AF_3L_200': CNN_AF_3L_200,
    # 'CNN_AF_3L_300': CNN_AF_3L_300,
    # 'CNN_AF_3L_700': CNN_AF_3L_700,
    # 'CNN_AF_3L_1400': CNN_AF_3L,
    # 'CNN_MF_1L': CNN_MF_1L,
    # 'CNN_MF_2L': CNN_MF_2L,
    # 'CNN_MF_3L': CNN_MF_3L,
}


def get_model(model_name: str, variables, num_classes: int, input_features: int, model_config: str, model_spec: str, with_input_batchnorm: bool):
    logger = getLogger("root.get_model")
    model_config_dict = {}
    if model_config and os.path.exists(model_config):
        logger.info(f"Parsing {model_config} to update {model_name} parameters")
        with open(model_config) as fp:
            model_config_dict = yaml.load(fp, Loader=yaml.CLoader)
    model_config_dict.update(dict(variables=variables, num_classes=num_classes, with_input_batchnorm=with_input_batchnorm, input_features=input_features))
    if model_name not in model_dict.keys():
        try:
            import tinyml_proprietary_models
            model_dict.update({model_name: tinyml_proprietary_models.get_model(model_name)})
        except ImportError:
            logger.info("tinyml_proprietary_models does not exist. Importing locally")
            if os.path.exists(model_spec):
                logger.info(f"Parsing {model_spec} to get {model_name} definition.")
                model_definition = misc_utils.import_file_or_folder(model_spec, __name__, force_import=True)
                model_dict.update({model_name: model_definition.get_model(model_name)})

    try:
       return model_dict[model_name](config=model_config_dict)
    except KeyError:
        traceback.print_exc()
        logger.error(f"{model_name} couldn't be found. If this is a protected model, it will only be available on the GUI version")
        raise RuntimeError(f"{model_name} couldn't be created.")
