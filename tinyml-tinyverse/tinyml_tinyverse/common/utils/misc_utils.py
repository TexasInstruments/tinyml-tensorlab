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
import importlib
import os
import sys


def str_or_bool(v):
    if isinstance(v, (str)):
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False
        elif v.lower() in ("none",):
            return None
        else:
            return v
      #
    else:
        return v


def str2bool_or_none(v):
    if v is None:
        return None
    elif isinstance(v, str) and v.lower() in ('none',):
        return None
    else:
        return str2bool(v)


def str2bool(v):
    '''a utility function used for argument parsing'''
    if v is None:
        return False
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'false', 'no', '0'):
            return False
        elif v.lower() in ('true', 'yes', '1'):
            return True
        #
    #
    return bool(v)


def import_file_or_folder(folder_or_file_name, package_name=None, force_import=False):
    if folder_or_file_name.endswith(os.sep):
        folder_or_file_name = folder_or_file_name[:-1]
    #
    if folder_or_file_name.endswith('.py'):
        folder_or_file_name = folder_or_file_name[:-3]
    #
    parent_folder = os.path.dirname(folder_or_file_name)
    basename = os.path.basename(folder_or_file_name)
    if force_import:
        sys.modules.pop(basename, None)
    #
    sys.path.insert(0, parent_folder)
    imported_module = importlib.import_module(basename, package_name or __name__)
    sys.path.pop(0)
    return imported_module