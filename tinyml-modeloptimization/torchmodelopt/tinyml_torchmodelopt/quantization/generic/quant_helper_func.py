import torch
from torch.fx import GraphModule, Node, symbolic_trace

import operator
from typing import Dict, List

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

def is_both_node_equal(named_modules: Dict, main_module_node: torch.Node, pattern_type_node: torch.Node) -> bool:
    '''Returns the truth value of both the given nodes'''
    both_node_equal = main_module_node.op == 'call_module' and isinstance(pattern_type_node, type) and isinstance(named_modules[main_module_node.target], pattern_type_node)
    both_node_equal = both_node_equal or (main_module_node.op == 'call_method' and isinstance(pattern_type_node, str) and main_module_node.target == pattern_type_node)
    both_node_equal = both_node_equal or (main_module_node.op == 'call_function' and are_both_function_equal(main_module_node.target, pattern_type_node))
    return both_node_equal

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
    named_modules = dict(main_module.named_modules())

    main_module_idx = 0
    matched_patterns = []

    while (main_module_idx < main_module_length):

        main_module_node = main_module_nodes[main_module_idx]
        nodes_matched = []

        curr_node = main_module_node

        if is_both_node_equal(named_modules, curr_node, pattern_type[0]):
            nodes_matched.append(curr_node)
            found_all = True
            
            for pattern_node in pattern_type[1:]:
                found = False
                for main_node in list(curr_node.users):
                    if is_both_node_equal(named_modules, main_node, pattern_node):
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

def compute_offset_scale_shift(offset: torch.Tensor, weight: torch.Tensor, round_offset: torch.Tensor=None, num_bits_shift: int=5, num_bits_scale: int=1, int_bias: bool=True, clip_weights: bool=False):
    """
    The functions takes quantization parameters (zero point, scale) to calculate the Offset, Scale, Shift required to quantize/dequantize the inputs. The range
    of weight must be less than equal to (2**num_bits_scale - 1). The tuple[torch.Tensor] output can be used to produce AMM (Offset, Scale, Right Shift) block
    using the class TINPUOffsetScaleShift(torch.nn.Module).

    Args:
        `offset`: additive offset
        `weight`: multiplicative weight
        `num_bits_shift`: number of bits to represent the shift value. this is not the number of bits to shift (which depends on the weight value), but the number of bits to represent the shift value.
        `num_bits_scale`: number of bits to represent the scale value.
        `int_bias`: if True, the offset is rounded to the nearest integer, otherwise it is left as a float
        `clip_weights`: if True, the weights are clipped to the range of [-(2**num_bits_scale - 1), (2**num_bits_scale - 1)], this will result in wrong quantization, but can be used for experiments

    Returns the computed offset, scale, shift

    Example:

        >>> oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(zero_point*0.0, 1/scale, num_bits_scale=8)
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127, ndim=4, dim=1)
    """
    one = torch.tensor([1.0])
    if not isinstance(offset, torch.Tensor):
        offset = one * (offset)
    if not isinstance(weight, torch.Tensor):
        weight = one * (weight)
    # Find the bounding values of scale
    scale_max = 2**num_bits_scale - 1
    # Max right shift operation supported
    shift_max = 2**num_bits_shift - 1
    # Separate the value and sign of weights
    if clip_weights:
        if max(weight.aminmax()) > scale_max:
            print("WARNING: compute_offset_scale_shift() - scaling out of range - will be clipped")
        #
        weight = weight.clip(-scale_max, scale_max)

    weight_abs = weight.abs()
    weight_sign = weight.sign()
    # Scale the weights
    weight_scale = torch.log2(scale_max/weight_abs)
    power_of_2 = torch.floor(weight_scale) * one
    shift = power_of_2.clamp(min=0, max=shift_max)

    scale_factor = torch.pow(one*2, shift)
    scaled_weights = scale_factor * weight_abs

    mask = torch.isnan(scaled_weights)
    scaled_weights[mask], shift[mask] = 0, 1

    if torch.sum(scaled_weights > scale_max) != 0:
        raise RuntimeError(
            f"Error in quantization.convert :: compute_offset_scale_shift. Scaling could not be converted.\n"
            f"Invalid scale values: {weight.cpu().detach().numpy()}\n"
            f"Maximum scale value allowed: {scale_max}\n"
            f"Make sure that the model is trained properly with good hyper parameters. "
            f"(Try adjusting: training epochs, learning rate, QAT after float training etc)\n"
            f"Check for multicollinearity and/or scaling input values.\n"
        )

    scaled_signed_weights = weight_sign * torch.round(scaled_weights)
    shift_mult = torch.pow(one*2, -shift)

    # add round offset to the offset. since the offset is before the scale, divide it by scale before adding
    shift_round_offset = torch.pow(one*2, (shift-1)) / scaled_signed_weights
    if round_offset is not None:
        shift_round_offset = round_offset
    offset = offset + shift_round_offset
    if int_bias:
        offset = torch.round(offset)
    return offset, scaled_signed_weights, shift_mult

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

def lint_and_recompile(main_module: GraphModule) -> None:
    ''' Lint and recompile the main_module '''
    main_module.graph.lint()
    main_module.recompile()
    return None

def get_name_from_module(node_module: torch.nn.Module, module_no: int) -> str:
    new_node_name = ''
    if hasattr(node_module, '__iter__'):
        for module in node_module:
            new_node_name += str(module.__class__.__name__) + '_'
    else:
        new_node_name += str(node_module.__class__.__name__) + '_'
    new_node_name += str(module_no)
    new_node_name = new_node_name.lower().replace('tinpuoffsetscaleshift', 'oss')
    return new_node_name

def remove_hanging_nodes(main_module: GraphModule) -> None:
    ''' Remove the hanging nodes from the main_module recursively '''
    while True:
        hanging_nodes = find_hanging_nodes(main_module)
        if len(hanging_nodes) == 0:
            break
        for node in hanging_nodes:
            main_module.graph.erase_node(node)

    lint_and_recompile(main_module)
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
    ''' The nodes from start to end is replaced with the replace module. All the intermediate 
    nodes are removed between start to end.

    Args:
        `main_module`: The graph module in which the replacement is to be done.
        `start`: The node where the replace module will be inserted.
        `end`: The node till where the nodes are to be removed.
        `replace_module`: The module which will be replaced with the start module.
        `module_no`: A number denoting the number of modules replaced till now. (Unique name for node)
    '''
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
            lint_and_recompile(main_module)
            return

    # Get the name of replaced module
    new_node_name = get_name_from_module(replace_module, module_no)
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
    lint_and_recompile(main_module)
    return None

def replace_call_module(main_module: GraphModule, start: Node, end: Node, replace_module: torch.nn.Module, module_no: int=0, rename_node_flag: bool=False) -> None:
    ''' The call module associated with the start node is replaced with the replace module. All the intermediate
    nodes are removed between start to end.

    Args:
        `main_module`: The graph module in which the replacement is to be done.
        `start`: The node having the call_module.
        `end`: The node till where the nodes are to be removed.
        `replace_module`: The module which will be replaced with the start module.
    '''
    main_modules = dict(main_module.named_modules())
    # Get the parent module name and attribute name
    parent_name, attr_name = _get_parent_name(start.target)
    parent_module = main_modules[parent_name]
    # Set the attribute of parent module with the replacement module
    parent_module.__setattr__(attr_name, replace_module)
    
    if rename_node_flag:
        new_node_name = get_name_from_module(replace_module, module_no)
        start.name = new_node_name

    # If there are more nodes between start and end, remove them all
    if start != end:
        # Initialize pointers for iteration
        new_node = start
        # Remove all the intermediate call module nodes
        users = list(start.users)
        for user in users:
            remove_intermediate_call_modules(main_module, new_node, user, end)
    remove_hanging_nodes(main_module)
    return None

def get_node_from_module(module: GraphModule, node: Node):
    named_modules = dict(module.named_modules())
    preserve_module = None
    if hasattr(node, 'target') and node.target in named_modules:
        preserve_module = named_modules[node.target]
    return preserve_module

def remove_node_from_module(module: GraphModule, node: Node):
    node_before = node.args[0]
    node.replace_all_uses_with(node_before)
    module.graph.erase_node(node)
    named_modules = dict(module.named_modules())
    if hasattr(node, 'target') and node.target in named_modules:
        module.delete_submodule(node.target)
    if node.name in module.graph._graph_namespace._used_names:
        module.graph._graph_namespace._used_names.remove(node.name)
    # Lint and recompile the graph and module
    lint_and_recompile(module)
    return None

def add_module_after_node(module: GraphModule, start: Node, end: Node, preserve_module: torch.nn.Module):
    with module.graph.inserting_after(end):
        # Add the submodule in module
        if hasattr(start, 'target'):
            module.add_submodule(start.target, preserve_module)
        # Add the module in graph
        new_node = module.graph.call_module(start.target)
        end.replace_all_uses_with(new_node)
        new_node.insert_arg(0, end) 
    return None

def add_node_after_node(module: GraphModule, start: Node, end: Node):
    # Get the module involved
    preserve_module = get_node_from_module(module, start)
    # Remove the node and replace it's use with previous node
    remove_node_from_module(module, start)
    # Add the preserved module after the end node
    add_module_after_node(module, start, end, preserve_module)
    # Lint and recompile the graph and module
    lint_and_recompile(module)
    return None

def add_activation_to_node(model: GraphModule, node: Node, range_max: int) -> None:
    f = getattr(model, str(node))
    if hasattr(f, "activation_post_process"):
        # Enforce the same quantization scale for all residual inputs
        f.activation_post_process.range_max = range_max
        f.activation_post_process.fixed_range = True
        setattr(model, str(node), f)
    return None

def set_quant_range(model: GraphModule, node: Node, target_name: str, quant_min: int, quant_max: int) -> None:
    modules_in_main_graph = dict(model.named_modules())
    f = getattr(model, str(node))
    # Check if the residual input is a ConvBn2d module
    condition_to_check = node.args[0].target in modules_in_main_graph
    condition_to_check = condition_to_check and target_name == operator.add 
    condition_to_check = condition_to_check and isinstance(modules_in_main_graph[node.args[0].target], torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn2d)
    if condition_to_check:
        # Set the quant_min and quant_max to a 10-bit signed range to prevent precision
        # loss with addition of 8-bit (signed or unsigned) residual data.
        if hasattr(f, "quant_min") and hasattr(f, "quant_max"):
            f.quant_min = quant_min
            f.quant_max = quant_max
            f.dtype = torch.qint32
            setattr(model, str(node), f)
    return None

def adjust_residual_inputs_qconfig(model : GraphModule, range_max: int=0, quant_min: int=torch.inf, quant_max: int=torch.inf) -> GraphModule:
    """
    Modify the QConfig inputs of add and concat operators to enforce the same quantization scale
    factors.
    model: the QAT prepared input model
    returns: the model with adjusted QConfigs
    """
    residual_operators = set([operator.add, torch.add, "add", torch.cat, torch.stack])
    # TODO: Try using a reference assignment for add.
    for node in model.graph.nodes:
        for n_child in node.users:
            target_name = n_child.target
            if target_name in residual_operators:
                if range_max != 0:
                    add_activation_to_node(model, node, range_max)
                if quant_min != torch.inf and quant_max != torch.inf:
                    set_quant_range(model, node, target_name, quant_min, quant_max)
    model.graph.lint()
    model.recompile()
    return model

def assign_same_observers(model : GraphModule, node_1: Node, node_2: Node) -> GraphModule:
    """
    Assigns the same observers to the residual inputs of a given node
    """
    activation_post_proc = None
    if hasattr(model, node_1.target):
        f = getattr(model, node_1.target)
        if hasattr(f, "activation_post_process"):
            activation_post_proc = f.activation_post_process

    if activation_post_proc and hasattr(model, node_2.target):
        f = getattr(model, node_2.target)
        setattr(f, "activation_post_process", activation_post_proc)
        model.graph.lint()
        model.recompile()
        
    return model

def assign_same_observers_for_residual_inputs(model: GraphModule):
    """
    Find the observers for the residual inputs of a given node and assign same observers to them
    """
    residual_operators = set([operator.add, torch.add, "add", torch.cat, torch.stack])
    for node in model.graph.nodes:
        target_name = node.target
        if target_name in residual_operators:
            node_1, node_2 = node.args
            assign_same_observers(model, node_1, node_2)    
    return    