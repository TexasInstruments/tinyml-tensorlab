#################################################################################
# Copyright (c) 2018-2025, Texas Instruments Incorporated - http://www.ti.com
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
from torch.fx import symbolic_trace

from . import replacer


def remove_identity(model: torch.nn.Module, verbose_mode=False, **kwargs):
    # removed due to RuntimeError
    # RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.
    # model=deepcopy(model)

    traced_model = symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    modules = dict(traced_model.named_modules())
    n = 0
    nodes = []
    for node in traced_model.graph.nodes:
        if (node.op == 'call_module') and isinstance(modules[node.target], torch.nn.Identity):
            nodes.append(node)
    for node in nodes:
        try:
            node.replace_all_uses_with(node.args[0])
            copy_found = False
            for node_1 in nodes:
                if node != node_1 and node.target == node_1.target:
                    copy_found = True
                    break
            if not copy_found:
                parent_name, name = replacer._get_parent_name(node.target)
                modules[parent_name].__delattr__(name)
                modules.pop(node.target)
            traced_model.graph.erase_node(node)
            n += 1
        except Exception as e:
            if verbose_mode:
                print(n, e)
    traced_model.graph.lint()
    traced_model.recompile()
    if verbose_mode:
        print('Identity removed', n)
    return traced_model
