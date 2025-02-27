import torch
from torch.fx import GraphModule, Node

from typing import Dict, List, Tuple
from .quant_modules import *

class TINPUQuantizedReplacementUtils():
    def __init__(self, model: GraphModule, weight_bw: int, activation_bw: int, power2_scale: bool):

        self.module: GraphModule = model
        self.graph_quant_params: Dict[str, Dict] = dict()
        self.module_num: int = 0

        self.weight_bw = weight_bw
        self.activation_bw = activation_bw
        self.num_bits_scale = 1 if power2_scale else 8
        self.rename_nodes_flag = False

        if self._check_module_before_quant():
            nodes = self._get_nodes()
            start_node, end_node = nodes[1], nodes[4]
            self.from_placeholder(start_node, end_node)

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

    # for the initial layers handling quantize_per_tensor
    def from_first_layer(self):
        first_quant_nodes = self._find_first_quant_node()

        for first_quant_node in first_quant_nodes:
            user = list(first_quant_node.users)[0]
            named_modules = self._get_named_modules()

            if is_both_node_equal(named_modules, user, torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d):
                self.from_q_qbn(first_quant_node, user)
            elif is_both_node_equal(named_modules, user, torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d):
                self.from_q_id(first_quant_node, user)
            elif is_both_node_equal(named_modules, user, torch.nn.Identity):
                self.from_q(first_quant_node, user)
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
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -(2**(self.activation_bw - 1)), 2**(self.activation_bw - 1) - 1, ndim=4, dim=1)
        # Replace quantize function with OSS Module
        replace_call_function_or_method(self.module, start, end, oss_module, self._get_module_num())
        return None

    def from_q_id(self, start: Node, end: Node):
        self.from_q(start, start)
        return None

    def from_q_qbn(self, start: Node, end: Node):
        # Quantized Batch Normalization Module
        qbn_module = self._get_named_modules()[end.target]
        bn_sigma = torch.sqrt(qbn_module.running_var + qbn_module.eps)

        scale = qbn_module.scale
        zero_point = qbn_module.zero_point

        oss_offset = (- qbn_module.running_mean + qbn_module.bias*bn_sigma)
        # first get the effective weight due to batchnorm
        combined_weight = (qbn_module.weight / bn_sigma)
        # then modify the weight by output scale so that the output is converted to output scale
        combined_weight = combined_weight / scale
        # OSS Module
        # for BN represented as offset, scale and shift, the scale can be an 8bit quantity
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(oss_offset, combined_weight, num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -2**(self.activation_bw - 1), 2**(self.activation_bw - 1) - 1, ndim=4, dim=1)
        # Remove the scale, zero_point, quantize method and bn layer with OSS Module
        replace_call_function_or_method(self.module, start, end, oss_module, self._get_module_num())
        return None
    
    def from_qbn(self, start: Node, end: Node):
        self.from_q_qbn(start, start)
        return None
    
    # Replacement Rules for quantized Convolution and Linear Layers
    def from_qconv_relu(self, start: Node, end: Node, with_relu: bool=True):
        qconvrelu_module = self._get_named_modules()[start.target]
        conv_module = torch.nn.Conv2d(qconvrelu_module.in_channels, qconvrelu_module.out_channels,
                                       kernel_size=qconvrelu_module.kernel_size, stride=qconvrelu_module.stride, 
                                       padding=qconvrelu_module.padding, dilation=qconvrelu_module.dilation, 
                                       groups=qconvrelu_module.groups, bias=False)
        
        weight = qconvrelu_module.weight()
        per_channel = (weight.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine))

        qweight = weight.data.detach().int_repr()
        conv_module.weight.data.copy_(qweight)

        weight_scale = weight.q_per_channel_scales() if per_channel else weight.q_scale()
        weight_zero_point = weight.q_per_channel_zero_points() if per_channel else weight.q_zero_point()
        
        input_scale, input_zero_point = self.get_q_params(start, using='prev')

        acc_scale = weight_scale * input_scale
        bias_scale = acc_scale
        bias_zero_point = weight_zero_point
        bias = qconvrelu_module.bias()

        # qbias = (torch.round(bias / bias_scale) + bias_zero_point).float()
        if per_channel:
            qbias = torch.quantize_per_channel(bias, bias_scale, bias_zero_point, 0, torch.qint32)
        else:
            qbias = torch.quantize_per_tensor(bias, bias_scale, bias_zero_point, 0, torch.qint32)
        
        qbias = qbias.int_repr()

        # conv_module.bias.data.copy_(qbias)
        relative_mult = (acc_scale / qconvrelu_module.scale).float()
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(qbias, relative_mult, num_bits_scale=self.num_bits_scale)
        
        if with_relu:
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -2**self.activation_bw + 1, 2**self.activation_bw - 1)
            seq_module = torch.nn.Sequential(conv_module, oss_module, torch.nn.ReLU(), torch.nn.Hardtanh(0, 2**self.activation_bw - 1))
        else:
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -2**self.activation_bw + 1, 2**self.activation_bw - 1)
            seq_module = torch.nn.Sequential(conv_module, oss_module)
        replace_call_module(self.module, start, end, seq_module, self._get_module_num(), self.rename_nodes_flag)
        return None

    def from_qconv(self, start: Node, end: Node, with_relu: bool=False):
        self.from_qconv_relu(start, start, with_relu)
        return None
    
    def from_qlinear(self, start: Node, end: Node, with_relu: bool=False):
        qlinear_module = self._get_named_modules()[start.target]
        linear_module = torch.nn.Linear(qlinear_module.in_features, qlinear_module.out_features, bias=False)

        weight = qlinear_module.weight()
        per_channel = (weight.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine))

        qweight = weight.data.detach().int_repr()
        linear_module.weight.data.copy_(qweight)

        weight_scale = weight.q_per_channel_scales() if per_channel else weight.q_scale()
        weight_zero_point = weight.q_per_channel_zero_points() if per_channel else weight.q_zero_point()

        input_scale, input_zero_point = self.get_q_params(start, using='prev')

        acc_scale = weight_scale * input_scale
        bias_scale = acc_scale
        bias_zero_point = weight_zero_point
        bias = qlinear_module.bias()

        # qbias = (torch.round(bias / bias_scale) + bias_zero_point).float()
        if per_channel:
            qbias = torch.quantize_per_channel(bias, bias_scale, bias_zero_point, 0, torch.qint32)
        else:
            qbias = torch.quantize_per_tensor(bias, bias_scale, bias_zero_point, 0, torch.qint32)
        
        qbias = qbias.int_repr()

        # conv_module.bias.data.copy_(qbias)
        relative_mult = (acc_scale / qlinear_module.scale).float()
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(qbias, relative_mult, num_bits_scale=self.num_bits_scale)

        if with_relu:
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 2**self.activation_bw - 1, ndim=2, dim=1)
            seq_module = torch.nn.Sequential(linear_module, oss_module, torch.nn.ReLU(), torch.nn.Hardtanh(0, 2**self.activation_bw - 1))
        else:
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -2**(self.activation_bw - 1), 2**(self.activation_bw - 1) - 1, ndim=2, dim=1)
            seq_module = torch.nn.Sequential(linear_module, oss_module)
        replace_call_module(self.module, start, end, seq_module, self._get_module_num(), self.rename_nodes_flag)
        return None

    def from_qlinear_relu(self, start: Node, end: Node):
        self.from_qlinear(start, end, with_relu=True)
        return None

    # Replacement Rules for Flatten
    def from_flatten(self, start: Node, end: Node):
        named_modules = self._get_named_modules()
        replace_module = named_modules[start.target]
        q_node = list(start.users)[-1]
        # Replaces the flatten module/flatten method with flatten module and drops the nodes till quantization node
        replace_call_module(self.module, start, q_node, replace_module, self._get_module_num(), self.rename_nodes_flag)
        return None

    def from_dq(self, start: Node, end: Node):
        # Identity Module
        id_module = torch.nn.Identity()
        # Replaces the dequantization method with Identity Module
        replace_call_function_or_method(self.module, start, end, id_module, self._get_module_num())
        return None
    
    def from_dq_flatten(self, start: Node, end: Node):
        # Replaces the dequantization method with Identity and removes
        # scale, zero_point, quantization node after flatten layer
        dq_node = start
        flatten_node = end
        self.from_dq(dq_node, dq_node)
        self.from_flatten(flatten_node, flatten_node)
        return None
    
    def from_add_relu(self, start: Node, end: Node, with_relu: bool=True):            
        scale, zero_point = self.get_q_params(start, using='prev')
        add_relu_block = AddReLUBlock(0, 2**self.activation_bw - 1, scale, zero_point*0.0, with_relu, num_bits_scale=self.num_bits_scale)
        replace_call_function_or_method(self.module, start, start, add_relu_block, self._get_module_num())
        return None
    
    def from_add(self, start: Node, end: Node, with_relu: bool=False):
        return self.from_add_relu(start, end, with_relu)
    
    def from_q_module(self, start: Node, end: Node):
        # Replaces the quantization method with OSS and removes
        # scale, zero_point, quantization node before flatten layer
        named_modules = self._get_named_modules()
        # Get the scale, zero_point of the quantization part
        scale = getattr(self.module, start.args[1].target)
        zero_point = getattr(self.module, start.args[2].target)
        # OSS Module
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(zero_point*0.0, 1/scale, num_bits_scale=self.num_bits_scale)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -(2**(self.activation_bw - 1)), 2**(self.activation_bw - 1) - 1, ndim=4, dim=1)
        # Get the module present after quantization
        if end.target in named_modules:
            # If flatten module is present in named_modules, we use the module
            replace_module = named_modules[end.target]
        # Sequential Module comprising of OSS and Replacement Module
        seq_module = torch.nn.Sequential(oss_module, replace_module)
        replace_call_function_or_method(self.module, start, end, seq_module, self._get_module_num())      
        return None
    
    def from_dq_with_dq(self, start: Node, end: Node):
        # Get the scale, zero_point from previous
        scale, zero_point = self.get_q_params(start, using='prev')
        id_module = torch.nn.Identity()
        mult_module = MultiplyModule(scale)
        # Sequential module comprising of identity and mult module
        seq_module = torch.nn.Sequential(id_module, mult_module)
        # Replaces dequantization method with sequential module
        replace_call_function_or_method(self.module, start, end, seq_module, self._get_module_num())
        return None

    # Replacement Rules for Pooling Layers
    def from_avg_pool2d(self, start: Node, end: Node):
        # AvgPool2D module
        pool_module = self._get_named_modules()[start.target]
        # Calculate the total_kernel_area from pool module kernel size
        total_kernel_area = pool_module.kernel_size[0] * pool_module.kernel_size[1]
        # OSS Module
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(torch.tensor((total_kernel_area+1)//2), torch.tensor(1 / total_kernel_area), num_bits_scale=self.num_bits_scale)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 2**self.activation_bw - 1, ndim=2, dim=1)
        # Multiply Module
        mult_module = MultiplyModule(total_kernel_area)
        # Round Module
        round_module = RoundModule()
        # Sequential Module comprising of AvgPool2D, Multiply, Round, OSS
        output_module = torch.nn.Sequential(pool_module, mult_module, round_module, oss_module)
        # Replace AvgPool2D with Sequential Module
        replace_call_module(self.module, start, end, output_module, self._get_module_num(), self.rename_nodes_flag)
        return None
    
    def from_adaptive_avg_pool2d(self, start: Node, end: Node):
        '''
        The below function offloads to NPU, but only works for output size -> (1, 1)
        Otherwise we will use a generic implementation, as AdaptiveAvgPooling will be 
        used mostly only at the model end and is not compute intensive
        '''
        # AdaptiveAvgPool2D Module
        pool_module = self._get_named_modules()[start.target]
        # Calculate the total_kernel_area from pool module output size
        total_kernel_area = pool_module.output_size[0] * pool_module.output_size[1]
        if total_kernel_area != 1:
            #  If output size isn't (1, 1), we will use the generic implementation
            replace_call_module(self.module, start, end, pool_module, self._get_module_num(), self.rename_nodes_flag)
            return None
        pool_module = AdaptiveAvgPool2d(activation_bw=self.activation_bw, num_bits_scale=self.num_bits_scale)
        # Replace AdaptiveAvgPool2D with Reduce, Round, OSS
        replace_call_module(self.module, start, end, pool_module, self._get_module_num(), self.rename_nodes_flag)
        return None

    def from_max_pool2d(self, start: Node, end: Node):
        pool_module = self._get_named_modules()[start.target]
        replace_call_module(self.module, start, end, pool_module, self._get_module_num(), self.rename_nodes_flag)
        return None