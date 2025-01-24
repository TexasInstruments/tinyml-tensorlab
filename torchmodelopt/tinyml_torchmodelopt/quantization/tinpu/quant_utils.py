import torch
from torch.fx import GraphModule, Node, symbolic_trace

from typing import Dict, List, Tuple

def are_both_function_equal(first_function, second_function) -> bool:
    ''' Returns the truth of value of operators of both the functions '''
    import operator
    operationDict = {torch.add: operator.add,torch.sub: operator.sub,torch.mul: operator.mul,
                        operator.add: torch.add,operator.sub: torch.sub,operator.mul: torch.mul}
    if first_function == second_function:
        return True
    elif hasattr(first_function, 'target') and first_function.target in operationDict.keys():
        # if it is one  of add, sub, mul from either of operator module or torch module it should be the counter part
        return second_function == operationDict[first_function]
    elif first_function in operationDict.keys():
        # if it is one  of add, sub, mul from either of operator module or torch module it should be the counter part
        return second_function == operationDict[first_function]
    else:
        return False

def simple_chain_searcher(main_module: GraphModule, pattern_type: List) -> List[torch.Node]:
    '''
    Finds the pattern_type in main_module graph and returns the list of nodes corresponding to pattern
    The function searches the list of main_module graph nodes in `linear` fashion. It matches the type of nodes 
    present in pattern_type or the target of main_module node.

    Args:
        `main_module`: The GraphModule in which the pattern is searched
        `pattern_type`: The List of types or target to match

    Returns the list of matched nodes
    '''
    main_module_nodes = list(main_module.graph.nodes)
    main_module_length = len(main_module_nodes)

    main_module_idx = 0
    matched_patterns = []

    def is_both_node_equal(main_module_node: torch.Node, pattern_type_node: torch.Node) -> bool:
        both_node_equal = main_module_node.op == 'call_module' and isinstance(pattern_type_node, type) and isinstance(dict(main_module.named_modules())[main_module_node.target], pattern_type_node)
        both_node_equal = both_node_equal or (main_module_node.op == 'call_method' and isinstance(pattern_type_node, str) and main_module_node.target == pattern_type_node)
        both_node_equal = both_node_equal or (main_module_node.op == 'call_function' and are_both_function_equal(main_module_node.target, pattern_type_node))
        return both_node_equal
    
    while (main_module_idx < main_module_length):

        main_module_node = main_module_nodes[main_module_idx]
        nodes_matched = []

        curr_node = main_module_node

        if is_both_node_equal(curr_node, pattern_type[0]):
            nodes_matched.append(curr_node)
            found_all = True
            
            for pattern_node in pattern_type[1:]:
                found = False
                for main_node in list(curr_node.users):
                    if is_both_node_equal(main_node, pattern_node):
                        curr_node = main_node
                        found = True
                        continue
                found_all = found_all and found
            if found_all:
                nodes_matched.append(curr_node)

        main_module_idx += 1

        if len(nodes_matched) == 2:
            matched_patterns.append(nodes_matched)

    return matched_patterns

def compute_offset_scale_shift(offset, weight, num_bits_shift=5, num_bits_scale=1, print_mse=False):
    """
    Represent offset, weight using add, mult and right shift
    :param offset: additive offset
    :param weight: multiplicative weight
    :param num_bits_shift: number of bits to represent the shift value. this is not the number of bits to shift (which depends on the weight value), but the number of bits to represent the shift value.
    :param num_bits_scale: number of bits to represent the scale value.
    :param print_mse: mean squared error occurred due to scale and shift
    :return: computed offset, scale, shift
    """
    one = torch.tensor([1.0])
    if not isinstance(offset, torch.Tensor):
        offset = one * (offset)
        weight = one * (weight)
    weight_abs = weight.abs()
    weight_sign = weight.sign()
    scale_max = (2**num_bits_scale)-1
    power_of_2 = torch.floor(torch.log2(scale_max/weight_abs))*one
    shift = power_of_2.clamp(min=0, max=((2**num_bits_shift)-1))
    scale = weight_abs * torch.pow(one*2, shift)

    mask = torch.isnan(scale)
    scale[mask] = 0
    shift[mask] = 1

    if torch.sum(scale > scale_max) != 0:
        raise RuntimeError(
            f"Error in Quant convert:compute_offset_scale_shift. Output multiplication could not be converted. \n"
            f"Invalid in output multiplication value: {weight.cpu().detach().numpy()} \n"
            f"Make sure that the model is trained properly with good hyper parameters. "
            f"(try adjusting: training epochs, learning rate, QAT after float training etc): \n"
        )

    scale = weight_sign*torch.round(scale)
    shift_mult = torch.pow(one*2, -shift)

    if print_mse:
        weight_hat = scale * torch.pow(one*2, -shift)
        mse = torch.mean((weight-weight_hat)**2)
        print(mse)

    # add round offset to the offset. since the offset is before the scale, divide it by scale before adding
    shift_round_offset = torch.pow(one*2, (shift-1)) / scale
    offset = torch.round(offset + shift_round_offset)
    return offset, scale, shift_mult

def _get_parent_name(target: str):
    ''' Gets the name of the parent module and attribute name of the module from the target of the module '''
    *parent, name = target.rsplit('.', 1)
    return (parent[0] if parent else ''), name

def find_hanging_nodes(main_module: GraphModule) -> List[Node]:
    ''' Returns a list of nodes which have no users and aren't placeholder or the output nodes '''
    count = []
    for node in main_module.graph.nodes:
        if (node.op not in ('output', 'placeholder') and len(node.users) == 0):
            count.append(node)
    return count

def remove_hanging_nodes(main_module: GraphModule) -> None:
    ''' Remove the hanging nodes from the main_module recursively '''
    while True:
        hanging_nodes = find_hanging_nodes(main_module)
        if len(hanging_nodes) == 0:
            break
        for node in hanging_nodes:
            main_module.graph.erase_node(node)

    main_module.graph.lint()
    main_module.recompile()

    return None

def remove_intermediate_call_modules(main_module: GraphModule, new_node: Node, start: Node, end: Node) -> None:
    ''' Removes all the call_modules and nodes present between start and end and replaces the uses with the new_node '''
    main_modules = dict(main_module.named_modules())
    ptr = start
    while ptr != end:
        if ptr.op == 'call_module':
            parent_name, name = _get_parent_name(ptr.target)
            parent_module = main_modules[parent_name]
            parent_module.__delattr__(name)
            
        users = list(ptr.users)

        for temp in users:
            ptr.replace_all_uses_with(new_node)
            main_module.graph.erase_node(ptr)
            ptr = temp

    if ptr.op == 'call_module':
        parent_name, name = _get_parent_name(end.target)
        parent_module = main_modules[parent_name]
        parent_module.__delattr__(name)

    ptr.replace_all_uses_with(new_node)
    main_module.graph.erase_node(end)
    return None

def replace_call_function_or_method(main_module: GraphModule, start: torch.Node, end: torch.Node, replace_module: torch.nn.Module, module_no: int=0) -> None:

    if start == end:
        traced_replacement = symbolic_trace(replace_module)
        replacement_nodes = [node for node in traced_replacement.graph.nodes if node.op not in ['placeholder', 'output']]

        if len(replacement_nodes) == 1:
            # call_function or call_method operation
            replacement_operation = replacement_nodes[0].op
            # function call or method name
            function_or_method = replacement_nodes[0].target
            # Replacing in main_module graph specifying insert point after start within this scope
            new_node = None
            with main_module.graph.inserting_after(start):
                # Insert a new node (replacement_node) using 'call_method' or 'call_function'
                new_node = getattr(main_module.graph, replacement_operation)(function_or_method, start.args, start.kwargs)
                # Replaces nodes that used the value of 'start' to now use that value new_node
                start.replace_all_uses_with(new_node)
            # Remove the unused 'start' node from graph as 'new_node' has replaced it
            main_module.graph.erase_node(start)
            main_module.graph.lint()
            main_module.recompile()
            return

    # Get the name of replaced module
    new_node_name = 'replaced_' + str(replace_module.__class__.__name__) + '_' + str(module_no)
    # Add the child module in main_module
    main_module.add_module(new_node_name, replace_module)

    # Inserting in main_module graph specifying insert point before start within this scope
    with main_module.graph.inserting_before(start):
        # Collect all the args which aren't attribute and needs to passed to call module
        args = []
        for arg in start.args:
            if type(arg) == Node and arg.op != "get_attr":
                args.append(arg)
        new_node = main_module.graph.call_module(new_node_name, tuple(args), {})
        # Remove all the intermediate call module nodes
        remove_intermediate_call_modules(main_module, new_node, start, end)
    main_module.graph.lint()
    main_module.recompile()
    return None

def replace_call_module(main_module: GraphModule, start: Node, end: Node, replace_module: torch.nn.Module, module_no: int=0) -> None:
    ''' The call module associated with the start node is replaced with the replace module. All the intermediate
    nodes are removed between start to end.

    Args:
    
        `main_module`: The graph module in which the replacement is to be done
        `start`: The node having the call_module
        `end`: The node till where the nodes are to be removed
        `replace_module`: The module which will be replaced with the start module
    '''
    main_modules = dict(main_module.named_modules())
    # Get the parent module name and attribute name
    parent_name, attr_name = _get_parent_name(start.target)
    parent_module = main_modules[parent_name]
    # Set the attribute of parent module with the replacement module
    parent_module.__setattr__(attr_name, replace_module)

    # If there are more nodes between start and end, remove them all
    if start != end:
        # Initialize pointers for iteration
        new_node = start
        # Remove all the intermediate call module nodes
        users = list(start.users)
        for user in users:
            remove_intermediate_call_modules(main_module, new_node, user, end)
    main_module.graph.lint()
    main_module.recompile()
    remove_hanging_nodes(main_module)
    return None

class ReduceSum(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=(2, 3))

class AdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round = RoundModule()
        self.reduce_sum = ReduceSum()

    def forward(self, x):
        shape = x.shape
        area = shape[2] * shape[3]

        offset, mult, shift_mult = compute_offset_scale_shift(1, 1 / area)
        oss = TINPUOffsetScaleShift(offset, mult, shift_mult, -128, 127, ndim=2, dim=1)
        
        x = self.reduce_sum(x) 
        x = self.round(x)         
        x = oss(x)     
        return x
    
class RoundModule(torch.nn.Module):
    def forward(self, x):
        return torch.round(x)
            
class MultiplyModule(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def forward(self, x):
        return torch.mul(x, self.value)

class AddReLUBlock(torch.nn.Module):
    def __init__(self, min_relu_clip, max_relu_clip, scale, zero_point, with_relu):
        super().__init__()
        self.with_relu = with_relu
        if with_relu:
            quant_min, quant_max = -max_relu_clip, max_relu_clip
        else:
            quant_min, quant_max = -128, 127

        offset, mult, shift_mult = compute_offset_scale_shift(zero_point, scale)
        self.oss = TINPUOffsetScaleShift(offset, mult, shift_mult, quant_min, quant_max)
        self.relu = torch.nn.ReLU()
        self.clip = torch.nn.Hardtanh(min_relu_clip, max_relu_clip)

    def forward(self, x, y):
        out = x + y
        y = self.oss(out)
        if self.with_relu:
            y = self.relu(y)
            y = self.clip(y)
        return y

class TINPUOffsetScaleShift(torch.nn.Module):
    def __init__(self, offset, mult, shift_mult, quant_min, quant_max, quantize_per_channel=False, use_floor=True, ndim=4, dim=1):
        super().__init__()
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.quantize_per_channel = quantize_per_channel
        self.use_floor = use_floor
        if ndim == 4 and dim == 1:
            self.register_buffer('offset', offset.reshape(1, -1, 1, 1))
            self.register_buffer('mult', mult.reshape(1, -1, 1, 1))
            self.register_buffer('shift_mult', shift_mult.reshape(1, -1, 1, 1))
        elif ndim == 2 and dim == 1:
            self.register_buffer('offset', offset.reshape(1, -1))
            self.register_buffer('mult', mult.reshape(1, -1))
            self.register_buffer('shift_mult', shift_mult.reshape(1, -1))
        elif ndim == 1:
            self.register_buffer('offset', offset.reshape(-1))
            self.register_buffer('mult', mult.reshape(-1))
            self.register_buffer('shift_mult', shift_mult.reshape(-1))
        else:
            raise RuntimeError('Invalid dimensions')

    def extra_repr(self):
        return f'offset={self.offset}, mult={self.mult}, shift={self.shift_mult}, quant_min={self.quant_min}, quant_max={self.quant_max}'

    def forward(self, x):
        y = (x + self.offset) * self.mult
        y = y * self.shift_mult
        if self.use_floor:
            # The floor operation mimics the actual shift and bit select in hardware
            y = torch.floor(y).clamp(min=self.quant_min, max=self.quant_max)
        else:
            y = torch.round(y).clamp(min=self.quant_min, max=self.quant_max)
        return y

class TINPUQuantizedReplacementUtils():
    def __init__(self, model: GraphModule):

        self.module: GraphModule = model
        self.graph_quant_params: Dict[str, Dict] = dict()
        self.module_num: int = 0

        if self._check_module_before_quant():
            nodes = self._get_nodes()
            start_node, end_node = nodes[1], nodes[4]
            self.from_placeholder(start_node, end_node)

        self._propagate_quant_params()

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
            prev_node: Node = self.graph_quant_params[node.name]['prev'][0]
            scale = self.graph_quant_params[prev_node.name]['scale']
            zero_point = self.graph_quant_params[prev_node.name]['zero_point']
        if using == 'this':
            scale = self.graph_quant_params[node.name]['scale']
            zero_point = self.graph_quant_params[node.name]['zero_point']
        if using == 'next':
            next_node: Node = self.graph_quant_params[node.name]['next'][0]
            scale = self.graph_quant_params[next_node.name]['scale']
            zero_point = self.graph_quant_params[next_node.name]['zero_point']
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
        named_modules = self._get_named_modules()
        # Get the nodes involved
        main_node, quant_node = start, end
        args_main_node = main_node.args
        # Get the name of the module and the module
        preserve_module = named_modules[main_node.target]
        # Remove and replace the module with the quantization nodes
        main_node.replace_all_uses_with(main_node.next)
        self.module.graph.erase_node(main_node)
        # Pass the args of main_node
        for idx, arg in enumerate(args_main_node):
            quant_node.update_arg(idx, arg)
        # Add the preserved module after the quantization node
        with self.module.graph.inserting_after(quant_node):
            # Add the submodule in module
            self.module.add_submodule(main_node.target, preserve_module)
            # Add the module in graph
            new_node = self.module.graph.call_module(main_node.target, tuple([quant_node]))
            quant_node.replace_all_uses_with(new_node)
            new_node.update_arg(0, quant_node) 
        # Lint and recompile the graph and module
        self.module.graph.lint()
        self.module.recompile()
        return None

    # Replacement Rules for quantized node at starting
    def from_q(self, start: Node, end: Node):
        # Quantization Node
        q_node = start
        scale = getattr(self.module, q_node.args[1].target)
        zero_point = getattr(self.module, q_node.args[2].target)
        # OSS Module
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(zero_point*0.0, 1/scale, num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127, ndim=4, dim=1)
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
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(oss_offset, combined_weight, num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127, ndim=4, dim=1)
        # Remove the scale, zero_point, quantize method and bn layer with OSS Module
        replace_call_function_or_method(self.module, start, end, oss_module, self._get_module_num())
        return None
    
    def from_qbn(self, start: Node, end: Node):
        self.from_q_qbn(start, start)
        return None
    
    # Replacement Rules for quantized Convolution and Linear Layers
    def from_qconv_relu(self, start: Node, end: Node, with_relu: bool=True):
        # zero_point_offset_for_activation = -128
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
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(qbias, relative_mult)
        
        if with_relu:
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -255, 255)
            seq_module = torch.nn.Sequential(conv_module, oss_module, torch.nn.ReLU(), torch.nn.Hardtanh(0, 255))
        else:
            # This clip is left to -255, 255 here, assuming that there is an Add and ReLU after this.
            # Otherwise it should be -128, 127
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -255, 255)
            seq_module = torch.nn.Sequential(conv_module, oss_module)
        replace_call_module(self.module, start, end, seq_module, self._get_module_num())
        return None

    def from_qconv(self, start: Node, end: Node, with_relu: bool=False):
        self.from_qconv_relu(start, start, with_relu)
        return None
    
    def from_qlinear(self, start: Node, end: Node, with_relu: bool=False):
        # zero_point_offset_for_activation = -128
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
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(qbias, relative_mult)

        if with_relu:
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 255, ndim=2, dim=1)
            seq_module = torch.nn.Sequential(linear_module, oss_module, torch.nn.ReLU(), torch.nn.Hardtanh(0, 255))
        else:
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127, ndim=2, dim=1)
            seq_module = torch.nn.Sequential(linear_module, oss_module)
        replace_call_module(self.module, start, end, seq_module, self._get_module_num())
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
        replace_call_module(self.module, start, q_node, replace_module, self._get_module_num())
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
        add_relu_block = AddReLUBlock(0, 255, scale, zero_point*0.0, with_relu)
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
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(zero_point*0.0, 1/scale, num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127, ndim=4, dim=1)
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
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(torch.tensor((total_kernel_area+1)//2), torch.tensor(1 / total_kernel_area), num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 255, ndim=2, dim=1)
        # Multiply Module
        mult_module = MultiplyModule(total_kernel_area)
        # Round Module
        round_module = RoundModule()
        # Sequential Module comprising of AvgPool2D, Multiply, Round, OSS
        output_module = torch.nn.Sequential(pool_module, mult_module, round_module, oss_module)
        # Replace AvgPool2D with Sequential Module
        replace_call_module(self.module, start, end, output_module, self._get_module_num())
        return None
    
    def from_adaptive_avg_pool2d(self, start: Node, end: Node):
        '''
        The below function offloads to NPU, but only works for output size->1,1
        Otherwise we will use a generic implementation, as AdaptiveAvgPooling will be 
        used mostly only at the model end and is not compute intensive
        '''
        # AdaptiveAvgPool2D Module
        pool_module = self._get_named_modules()[start.target]
        # Calculate the total_kernel_area from pool module output size
        total_kernel_area = pool_module.output_size[0] * pool_module.output_size[1]
        if total_kernel_area != 1:
            #  If output size isn't (1, 1), we will use the generic implementation
            return self.from_passthrough_module(start, end)
        pool_module = AdaptiveAvgPool2d()
        # Replace AdaptiveAvgPool2D with Reduce, Round, OSS
        replace_call_module(self.module, start, end, pool_module, self._get_module_num())
        return None

    def from_max_pool2d(self, start: Node, end: Node):
        named_modules = self._get_named_modules()
        replace_module = named_modules[start.target]
        replace_call_module(self.module, start, end, replace_module, self._get_module_num())
        return None

    def from_passthrough_module(self, start: Node, end: Node):
        named_modules = self._get_named_modules()
        replace_module = named_modules[start.target]
        replace_call_module(self.module, start, end, replace_module, self._get_module_num())
        return None