#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

import torch
from torch import nn

from copy import deepcopy
from types import FunctionType
from typing import Union, Dict, Any
import warnings
from .surgery import _replace_unsupported_layers

def convert_to_lite_fx(model: torch.nn.Module, replacement_dict: Dict[Any, Union[torch.nn.Module, callable]] = None, example_inputs: list = None, example_kwargs: dict = None, verbose_mode: bool = False, **kwargs):
    '''
    converts model into lite model using replacement dict
    if no replacement dict is provided it does the default replacement
    '''
    # "example_inputs optional and used only in models using LayerNorm. Using a default value since it was not provided.
    example_inputs = example_inputs if example_inputs is not None else torch.rand(1, 3, 224, 224)  # Default input shape
    example_kwargs = example_kwargs or {}
    return replace_unsupported_layers(model, example_inputs=example_inputs, example_kwargs=example_kwargs, replacement_dict=replacement_dict, verbose_mode=verbose_mode, **kwargs)


# Default Flags for replacement dict
# for custom replacement add a custom flag name (any string not in default flag) as key and map it to a dict containing pattern and replacement
# note if same key is used for two pattern the last replacement will be performed
default_replacement_flag_dict = {
}

# Default Flags for replacement dict with no training required as subset of default flags
# for custom replacement add a custom flag name (any string not in default flag) as key and map it to a dict containing pattern and replacement
# note if same key is used for two pattern the last replacement will be performed
default_replacement_flag_dict_no_training = {
}

# Mapping between the flags and the actual replacements corresponding to them
# This dictionary is used whenever a flag is enabled to fetch the corresponding replacement entries
flag_to_dict_entries = {
}


# returns default dictionary for replacement
def get_replacement_flag_dict_default(return_flags=True, can_retrain=True):
    '''
    returns the default flag dictionary.
    to see the dict print 'default_replacement_flag_dict' from the file this function is in
    '''
    flag_dict = default_replacement_flag_dict if can_retrain else default_replacement_flag_dict_no_training
    if return_flags:
        return flag_dict
    replacement_entries_dict = {}
    for k, v in flag_dict.items():
        if k in flag_to_dict_entries and v in (True, False):
            if v:
                v = flag_to_dict_entries[k]
            else:
                continue
        else:
            continue
        replacement_entries_dict.update({k, v})
    return replacement_entries_dict


def get_replacement_dict(
        replacement_flag_dict: dict[
            str | nn.Module | FunctionType | type, bool | nn.Module | FunctionType | type | tuple[
                FunctionType, FunctionType]] = None,
        can_retrain: bool = True,
):
    '''
    this function actually converts the flags mapped to True to their corresponding replacements
    if no replacement_flag_dict is given it uses default flag dictionary based on can_retrain
    if the flags is not registered in 'flag_to_dict_entries', its value should be a dict of replacement and that will be updated in the dictionary
    '''

    if can_retrain:
        replacement_flag_dict = replacement_flag_dict or default_replacement_flag_dict
    else:
        replacement_flag_dict = replacement_flag_dict or default_replacement_flag_dict_no_training

    replacement_dict: dict[Any, list[tuple]] = {}

    replacement_dict = {}
    for k, v in replacement_flag_dict.items():
        if k in flag_to_dict_entries and v in (True, False):
            if v:
                v = flag_to_dict_entries[k]
            else:
                continue
        else:
            if not isinstance(v, dict):
                warnings.warn(
                    f'if {k} is not a default flag or its value is not a boolean, the value must be a dict. So, this entry will be discarded!')
                continue
        replacement_dict.update(v)
    return replacement_dict


def replace_unsupported_layers(model: nn.Module, example_inputs: list = None, example_kwargs: dict = None,
                               replacement_dict: Dict[Any,
                                                      Union[nn.Module, callable]] = None,
                               copy_args: list = [], can_retrain=True, verbose_mode: bool = False, **kwargs):
    # TODO write appropriate documentation for this function
    '''
    wrapper to the function that does the surgery

    it does default surgery if no replacement dictionary is given
    replacement dictionary must contains flag name as keys and True/False or a replacement dictionary corresponding to flag as value

    behavior for each value:
    value               behavior
    True            ->  convert according to mapped replacement dict
    False           ->  discard it
    dict            ->  update the main replacement dictionary with the entries of it

    values for replacement dict
    keys                value
    callable        ->  callable            : any call function to call_function if they take same argument partial argument may be used
    callable        ->  nn.Module           : any call function to call_function if they take same argument partial argument may be used
    Any             ->  Callable            : any self-made surgery function
    nn.Module       ->  nn.Module/type      : any nn.Module pattern to replace with another nn.Module
    type            ->  type/nn.Module      : replaces sub-module of same type as pattern using traditional python approach
    '''
    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
    # TODO Check for functions
    if model.training:
        RuntimeWarning(
            "The model is in train mode, converting to eval mode. This might change the network behavior.")
        model.eval()
        is_train_mode = True
    else:
        is_train_mode = False

    replacement_dict = get_replacement_dict(
        replacement_dict, can_retrain=can_retrain)

    model = deepcopy(model)

    final_model = surgery._replace_unsupported_layers(model, example_inputs, example_kwargs, replacement_dict, copy_args,
                                                      verbose_mode, **kwargs)

    if is_train_mode:
        final_model.train()

    return final_model
