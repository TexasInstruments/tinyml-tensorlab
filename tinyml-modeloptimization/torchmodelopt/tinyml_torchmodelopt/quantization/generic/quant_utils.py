import torch
from torch.fx import GraphModule, Node

from typing import Dict, List, Tuple
from .quant_helper_func import *
from .quant_modules import *

class GENERICQuantizedReplacementUtils():
    def __init__(self, model: GraphModule, weight_bw: int, activation_bw: int, power2_scale: bool):

        self.module: GraphModule = model
        self.graph_quant_params: Dict[str, Dict] = dict()
        self.module_num: int = 0

        self.weight_bw = weight_bw
        self.activation_bw = activation_bw
        self.num_bits_scale = 1 if power2_scale else 8
        self.rename_nodes_flag = False

        self._propagate_quant_params()
        if self.rename_nodes_flag:
            self.rename_nodes()
        self.from_first_layer()

    def _get_nodes(self) -> List[Node]:
        return list(self.module.graph.nodes)
    
    def _get_named_modules(self) -> Dict:
        return dict(self.module.named_modules())

    def _get_module_num(self, update: bool=True) -> int:
        if update:
            self.module_num += 1
        return self.module_num
    
    def _check_module_before_quant(self) -> bool:
        nodes = self._get_nodes()
        # Checks if there is a module before quantize_per_tensor and after placeholder
        placeholder_node = nodes[0]
        for user in placeholder_node.users:
            if user.op == 'call_module':
                return True
        return False
    
    def rename_nodes(self):
        nodes = self._get_nodes()
        named_modules = self._get_named_modules()
        count = 0
        for node in nodes:
            if node.op == 'call_module':
                new_node_name = ''
                node_module = named_modules[node.target]
                if hasattr(node_module, '__iter__'):
                    for module in node_module:
                        new_node_name += module.__class__.__name__ + '_'
                else:
                    new_node_name = node_module.__class__.__name__
                node.name = new_node_name.lower() + str(count)
                count += 1
        return None
    
    def _propagate_quant_params(self) -> None:
        for node in self.module.graph.nodes:
            scale, zero_point = 1.0, 0.0

            if node.op == 'placeholder':                                            # Inputs (x) nodes
                scale, zero_point = 1.0, 0.0
            elif node.op == 'call_module':                                          # ConvBnRelu, AvgPool, Linear, Flatten nodes
                named_modules = self._get_named_modules()
                module = named_modules[node.target]
                if hasattr(module, 'scale') and hasattr(module, 'zero_point'):     
                    # ConvBnRelu, Linear modules already have q_params
                    scale, zero_point = module.scale, module.zero_point
                else:                                                               
                    # Flatten, Pooling modules have single input (args)
                    args = node.args[0]
                    scale = self.graph_quant_params[args.name]['scale']
                    zero_point = self.graph_quant_params[args.name]['zero_point']
            elif node.op == 'call_method':                                          # Dequantize node
                # Quant params from previous node
                if node.name.startswith(('dequantize')):
                    scale = 1.0
                    zero_point = 0.0
                else:
                    prev_node = node.args[0]
                    scale = self.graph_quant_params[prev_node.name]['scale']
                    zero_point = self.graph_quant_params[prev_node.name]['zero_point']
            elif node.op == 'call_function':                                        # Quantize_per_tensor, Reshape args nodes
                # Get the quant params from model[arg_node.target]
                if node.name.startswith(('quantize_per_tensor')):
                    args = node.args
                    scale = getattr(self.module, args[1].target)
                    zero_point = getattr(self.module, args[2].target)
                elif node.name.startswith(('add')):
                    args = node.args
                    scale = getattr(self.module, args[2].target)
                    zero_point = getattr(self.module, args[3].target)
                else:
                    scale = self.graph_quant_params[node.prev.name]['scale']
                    zero_point = self.graph_quant_params[node.prev.name]['zero_point']
            elif node.op == 'get_attr':                                              # Attribute nodes of functions
                scale, zero_point = 1.0, 0.0
            elif node.op == 'output':                                                # Output node
                scale, zero_point = 1.0, 0.0
            # Add the quantization params of the current node
            if node.name not in self.graph_quant_params.keys():
                self.graph_quant_params[node.name] = dict()
            self.graph_quant_params[node.name]['scale'] = scale
            self.graph_quant_params[node.name]['zero_point'] = zero_point
            # Add the quantization params for the users of current node
            for user in node.users:
                if 'next' in self.graph_quant_params[node.name]:
                    self.graph_quant_params[node.name]['next'].append(user)
                else:
                    self.graph_quant_params[node.name]['next'] = [user]
                if user.name not in self.graph_quant_params.keys():
                    self.graph_quant_params[user.name] = dict()
                if 'prev' in self.graph_quant_params[user.name]:
                    self.graph_quant_params[user.name]['prev'].append(node)
                else:
                    self.graph_quant_params[user.name]['prev'] = [node]
        return None
    
    def get_q_params(self, node: Node, using: str='prev') -> Tuple[float]:
        scale, zero_point = 1.0, 0.0
        if using == 'prev':
            node_name = node.name
            if self.rename_nodes_flag and node.op != 'call_method':
                node_name = node.target.replace('.', '_')
            prev_node: Node = self.graph_quant_params[node_name]['prev'][0]
            if isinstance(prev_node.target, str) and self.rename_nodes_flag:
                prev_node_name = prev_node.target.replace('.', '_')
            else:
                prev_node_name = prev_node.name
            scale = self.graph_quant_params[prev_node_name]['scale']
            zero_point = self.graph_quant_params[prev_node_name]['zero_point']
        if using == 'this':
            node_name = node.name
            if self.rename_nodes_flag:
                node_name = node.target.replace('.', '_')
            scale = self.graph_quant_params[node_name]['scale']
            zero_point = self.graph_quant_params[node_name]['zero_point']
        if using == 'next':
            node_name = node.name
            if self.rename_nodes_flag and node.op != 'call_method':
                node_name = node.target.replace('.', '_')
            next_node: Node = self.graph_quant_params[node_name]['next'][0]
            if isinstance(next_node.target, str):
                next_node_name = next_node.target.replace('.', '_')
            else:
                next_node_name = next_node.name
            scale = self.graph_quant_params[next_node_name]['scale']
            zero_point = self.graph_quant_params[next_node_name]['zero_point']
        return (scale, zero_point)
    
    def update_module(self, module: GraphModule) -> GraphModule:
        remove_hanging_nodes(self.module)
        module = self.module
        return module
    
    def search_pattern(self, replacement_pattern: List) -> List[Node]:
        matches = simple_chain_searcher(self.module, replacement_pattern)
        return matches
    
    # Special Replacement rule for quantization after module at start
    def from_placeholder(self, start: Node, end: Node) -> None:
        main_node, quant_node = start, end
        add_node_after_node(self.module, main_node, quant_node)
        return None

    def _find_first_quant_node(self):
        first_quant_node = []
        nodes = self._get_nodes()
        named_modules = self._get_named_modules()

        placeholder_node = nodes[0]
        # add the placeholder node in bfs queue
        bfs = [placeholder_node]

        while bfs.__len__() != 0:
            node = bfs[0]
            bfs.pop(0)
            # check if the node is a quant node, if not 
            # find where it is present recursively after placeholder
            if is_both_node_equal(named_modules, node, torch.quantize_per_tensor):
                first_quant_node.append(node)
            else:
                bfs += list(node.users)
        return first_quant_node
    

    # for the initial layers handling quantize_per_tensor
    def from_first_layer(self):
        first_quant_nodes = self._find_first_quant_node()

        for first_quant_node in first_quant_nodes:
            user = list(first_quant_node.users)[0]
            named_modules = self._get_named_modules()

            if is_both_node_equal(named_modules, user, torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d):
                self.from_q_qbn(first_quant_node, user)
            else:
                pass
        return None
    
    # Replacement Rules for quantized node at starting
    def from_q(self, start: Node, end: Node):
        # Quantization Node
        q_node = start
        scale = getattr(self.module, q_node.args[1].target)
        zero_point = getattr(self.module, q_node.args[2].target)
        # OSS Module
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(zero_point*0.0, 1/scale, num_bits_scale=8)
        oss_module = GENERICOffsetScaleShift(oss_offset, oss_scale, oss_shift, 1.0, 0, -(2**(self.activation_bw - 1)), 2**(self.activation_bw - 1) - 1, ndim=4, dim=1)
        # Replace quantize function with OSS Module
        replace_call_function_or_method(self.module, start, end, oss_module, self._get_module_num())
        return None

    def from_q_id(self, start: Node, end: Node):
        self.from_q(start, start)
        return None
    
    def from_qbn(self, start: Node, end: Node):
        # Quantized Batch Normalization Module
        qbn_module = self._get_named_modules()[end.target]
        bn_sigma = torch.sqrt(qbn_module.running_var + qbn_module.eps)

        scale = qbn_module.scale
        zero_point = qbn_module.zero_point

        bn_offset = (- qbn_module.running_mean + qbn_module.bias * bn_sigma / qbn_module.weight)
        # first get the effective weight due to batchnorm
        combined_weight = (qbn_module.weight / bn_sigma)
        # then modify the weight by output scale so that the output is converted to output scale
        bn_scale = combined_weight
        # OSS Module
        # for BN represented as offset, scale and shift, the scale can be an 8bit quantity        
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(bn_offset, bn_scale, num_bits_scale=8)
        normalize_input = GENERICOffsetScaleShift(oss_offset, oss_scale, oss_shift, scale, zero_point, -2**(self.activation_bw - 1), 2**(self.activation_bw - 1) - 1, ndim=4, dim=1)
        # Remove the scale, zero_point, quantize method and bn layer with OSS Module
        replace_call_function_or_method(self.module, start, end, normalize_input, self._get_module_num())
        return None

    def from_q_qbn(self, start: Node, end: Node):
        # Quantized Batch Normalization Module
        qbn_module = self._get_named_modules()[end.target]
        bn_sigma = torch.sqrt(qbn_module.running_var + qbn_module.eps)

        scale = qbn_module.scale
        zero_point = qbn_module.zero_point

        bn_offset = (- qbn_module.running_mean + qbn_module.bias * bn_sigma / qbn_module.weight)
        # first get the effective weight due to batchnorm
        combined_weight = (qbn_module.weight / bn_sigma)
        # then modify the weight by output scale so that the output is converted to output scale
        bn_scale = combined_weight
        # OSS Module
        # for BN represented as offset, scale and shift, the scale can be an 8bit quantity        
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(bn_offset, bn_scale, num_bits_scale=8)
        normalize_input = GENERICOffsetScaleShift(oss_offset, oss_scale, oss_shift, scale, zero_point, -2**(self.activation_bw - 1), 2**(self.activation_bw - 1) - 1, ndim=4, dim=1)
        qbn_module = torch.nn.Sequential(normalize_input)
        # Remove the scale, zero_point, quantize method and bn layer with OSS Module
        replace_call_function_or_method(self.module, start, end, qbn_module, self._get_module_num())
        return None
    
    def from_permute(self, start: Node, end: Node):
        q_node = start.args[0]
        scale = getattr(self.module, q_node.args[1].target)
        zp = getattr(self.module, q_node.args[2].target)
        replace_call_function_or_method(self.module, start, end, PermBlock(scale, zp), self._get_module_num())
        return None