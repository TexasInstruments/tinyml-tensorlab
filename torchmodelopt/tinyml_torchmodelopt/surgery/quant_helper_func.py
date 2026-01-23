import torch
from torch.fx import GraphModule, Node, symbolic_trace

import operator
from typing import Dict, List

def are_both_function_equal(first_function, second_function) -> bool:
    """Check if two functions are equivalent operators.
    
    Returns:
        bool: True if functions are equivalent.
    """
    operation_dict = {torch.add: operator.add, torch.sub: operator.sub, torch.mul: operator.mul,
                      operator.add: torch.add, operator.sub: torch.sub, operator.mul: torch.mul}
    if first_function == second_function:
        return True
    elif hasattr(first_function, 'target') and first_function.target in operation_dict:
        return second_function == operation_dict[first_function]
    elif first_function in operation_dict:
        return second_function == operation_dict[first_function]
    else:
        return False

def is_both_node_equal(named_modules: Dict, main_module_node: torch.Node, pattern_type_node: torch.Node) -> bool:
    """Check if two nodes are equivalent.
    
    Args:
        named_modules: Dictionary of module names to modules
        main_module_node: Node from main module
        pattern_type_node: Pattern node to compare
        
    Returns:
        bool: True if nodes are equivalent
    """
    is_equal = main_module_node.op == 'call_module' and isinstance(pattern_type_node, type) and isinstance(named_modules[main_module_node.target], pattern_type_node)
    is_equal = is_equal or (main_module_node.op == 'call_method' and isinstance(pattern_type_node, str) and main_module_node.target == pattern_type_node)
    is_equal = is_equal or (main_module_node.op == 'call_function' and are_both_function_equal(main_module_node.target, pattern_type_node))
    return is_equal

def simple_chain_searcher(main_module: GraphModule, pattern_type: List) -> List[List[torch.Node]]:
    """Find patterns in the module graph and return matching node sequences.
    
    Searches through main_module graph nodes linearly, matching types or targets
    present in pattern_type.

    Args:
        main_module: GraphModule in which the pattern is searched
        pattern_type: List of types or targets to match

    Returns:
        List of matched node sequences
    """
    module_nodes = list(main_module.graph.nodes)
    module_length = len(module_nodes)
    named_modules = dict(main_module.named_modules())

    node_idx = 0
    matched_patterns = []

    while node_idx < module_length:

        current_node = module_nodes[node_idx]
        matched_nodes = []

        curr_node = current_node

        if is_both_node_equal(named_modules, curr_node, pattern_type[0]):
            matched_nodes.append(curr_node)
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
                matched_nodes.append(curr_node)

        node_idx += 1

        if len(matched_nodes) == 2:
            matched_patterns.append(matched_nodes)

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
    one = torch.ones(size=(1,))
    if not isinstance(offset, torch.Tensor):
        offset = one * (offset)
    if not isinstance(weight, torch.Tensor):
        weight = one * (weight)
    # Find the bounding values of scale
    scale_max: int = 2**num_bits_scale - 1
    # Max right shift operation supported
    shift_max: int = 2**num_bits_shift - 1
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
    import warnings
    from torch.jit import TracerWarning
    warnings.filterwarnings("ignore", category=TracerWarning)

    if torch.any(scaled_weights > scale_max):
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
    """Extract parent module name and attribute name from a target string.
    
    Args:
        target: Target string in format 'parent.module.name'
        
    Returns:
        Tuple of (parent_name, attribute_name)
    """
    *parent, name = target.rsplit('.', 1)
    return (parent[0] if parent else ''), name

def find_hanging_nodes(main_module: GraphModule) -> List[Node]:
    """Find nodes with no users that aren't placeholders or outputs.
    
    Args:
        main_module: GraphModule to search
        
    Returns:
        List of hanging nodes
    """
    hanging_nodes = []
    for node in main_module.graph.nodes:
        if node.op not in ('output', 'placeholder') and len(node.users) == 0:
            hanging_nodes.append(node)
    return hanging_nodes

def lint_and_recompile(main_module: GraphModule) -> None:
    """Lint and recompile the graph module.
    
    Args:
        main_module: GraphModule to lint and recompile
    """
    main_module.graph.lint()
    main_module.recompile()

def get_name_from_module(node_module: torch.nn.Module, module_no: int) -> str:
    """Generate a unique name from a module.
    
    Args:
        node_module: Module to generate name from
        module_no: Module number for uniqueness
        
    Returns:
        Generated module name (snake_case)
    """
    module_name = ''
    if hasattr(node_module, '__iter__'):
        for module in node_module:
            module_name += str(module.__class__.__name__) + '_'
    else:
        module_name += str(node_module.__class__.__name__) + '_'
    module_name += str(module_no)
    module_name = module_name.lower().replace('tinpuoffsetscaleshift', 'oss')
    return module_name

def remove_hanging_nodes(main_module: GraphModule) -> None:
    """Remove hanging nodes from the module recursively.
    
    Args:
        main_module: GraphModule to clean
    """
    while True:
        hanging_nodes = find_hanging_nodes(main_module)
        if not hanging_nodes:
            break
        for node in hanging_nodes:
            main_module.graph.erase_node(node)

    lint_and_recompile(main_module)

def remove_intermediate_call_modules(main_module: GraphModule, new_node: Node, start: Node, end: Node) -> None:
    """Remove call_module nodes between start and end nodes.
    
    Removes all intermediate nodes and replaces uses with new_node.
    
    Args:
        main_module: GraphModule to modify
        new_node: Node to replace with
        start: Starting node
        end: Ending node
    """
    named_modules = dict(main_module.named_modules())
    ptr = start
    while ptr != end:
        if ptr.op == 'call_module':
            parent_name, attr_name = _get_parent_name(ptr.target)
            parent_module = named_modules[parent_name]
            parent_module.__delattr__(attr_name)
            
        users = list(ptr.users)

        for temp in users:
            ptr.replace_all_uses_with(new_node)
            main_module.graph.erase_node(ptr)
            ptr = temp

    if ptr.op == 'call_module':
        parent_name, attr_name = _get_parent_name(end.target)
        parent_module = named_modules[parent_name]
        parent_module.__delattr__(attr_name)

    ptr.replace_all_uses_with(new_node)
    main_module.graph.erase_node(end)

def replace_call_function_or_method(main_module: GraphModule, start: torch.Node, end: torch.Node, replace_module: torch.nn.Module, module_no: int = 0) -> None:
    """Replace nodes between start and end with a replacement module.
    
    Removes intermediate nodes and inserts the replacement module.

    Args:
        main_module: Graph module to modify
        start: Node where replacement will be inserted
        end: Final node in replacement range
        replace_module: Module to insert
        module_no: Module number for unique naming
    """
    if start == end:
        traced_replacement = symbolic_trace(replace_module)
        replacement_nodes = [node for node in traced_replacement.graph.nodes if node.op not in ['placeholder', 'output']]

        if len(replacement_nodes) == 1:
            # Get operation type and target from replacement node
            replacement_operation = replacement_nodes[0].op
            function_or_method = replacement_nodes[0].target
            new_node = None
            with main_module.graph.inserting_after(start):
                # Insert new node using the operation type (call_method or call_function)
                new_node = getattr(main_module.graph, replacement_operation)(function_or_method, start.args, start.kwargs)
                # Replace all uses of start node with new node
                start.replace_all_uses_with(new_node)
            # Remove the original start node
            main_module.graph.erase_node(start)
            lint_and_recompile(main_module)
            return

    # Get the name of replaced module
    new_node_name = get_name_from_module(replace_module, module_no)
    # Add the module to main_module
    main_module.add_module(new_node_name, replace_module)

    # Insert replacement in graph before start node
    with main_module.graph.inserting_before(start):
        # Collect non-attribute arguments to pass to call module
        args = []
        for arg in start.args:
            if isinstance(arg, Node) and arg.op != "get_attr":
                args.append(arg)
        new_node = main_module.graph.call_module(new_node_name, tuple(args), {})
        # Remove intermediate nodes
        remove_intermediate_call_modules(main_module, new_node, start, end)
    lint_and_recompile(main_module)

def replace_call_module(main_module: GraphModule, start: Node, end: Node, replace_module: torch.nn.Module, module_no: int = 0, rename_node_flag: bool = False) -> None:
    """Replace the call_module at start node with replace_module.
    
    Removes all intermediate nodes between start and end.

    Args:
        main_module: Graph module to modify
        start: Node with call_module to replace
        end: Final node in replacement range
        replace_module: Module to replace with
        module_no: Module number for unique naming
        rename_node_flag: Whether to rename the node
    """
    named_modules = dict(main_module.named_modules())
    # Get parent module and attribute name
    parent_name, attr_name = _get_parent_name(start.target)
    parent_module = named_modules[parent_name]
    # Replace the module attribute with replacement
    parent_module.__setattr__(attr_name, replace_module)
    
    if rename_node_flag:
        new_node_name = get_name_from_module(replace_module, module_no)
        start.name = new_node_name

    # Remove intermediate nodes between start and end
    if start != end:
        new_node = start
        # Remove all intermediate call module nodes
        users = list(start.users)
        for user in users:
            remove_intermediate_call_modules(main_module, new_node, user, end)
    remove_hanging_nodes(main_module)

def get_node_from_module(module: GraphModule, node: Node):
    """Get the module associated with a node.
    
    Args:
        module: GraphModule to search
        node: Node to get module from
        
    Returns:
        Module associated with node or None
    """
    named_modules = dict(module.named_modules())
    preserve_module = None
    if hasattr(node, 'target') and node.target in named_modules:
        preserve_module = named_modules[node.target]
    return preserve_module

def remove_node_from_module(module: GraphModule, node: Node):
    """Remove a node from the module graph.
    
    Args:
        module: GraphModule to modify
        node: Node to remove
    """
    node_before = node.args[0]
    node.replace_all_uses_with(node_before)
    module.graph.erase_node(node)
    named_modules = dict(module.named_modules())
    if hasattr(node, 'target') and node.target in named_modules:
        module.delete_submodule(node.target)
    if node.name in module.graph._graph_namespace._used_names:
        module.graph._graph_namespace._used_names.remove(node.name)
    # Lint and recompile the graph
    lint_and_recompile(module)

def add_module_after_node(module: GraphModule, start: Node, end: Node, preserve_module: torch.nn.Module):
    """Add a module after a specified node in the graph.
    
    Args:
        module: GraphModule to modify
        start: Starting node
        end: Node after which to insert
        preserve_module: Module to add
    """
    with module.graph.inserting_after(end):
        # Add the submodule to the module
        if hasattr(start, 'target'):
            module.add_submodule(start.target, preserve_module)
        # Add the call_module node to the graph
        new_node = module.graph.call_module(start.target)
        end.replace_all_uses_with(new_node)
        new_node.insert_arg(0, end)

def add_node_after_node(module: GraphModule, start: Node, end: Node):
    """Add a node after a specified node by preserving and reinserting the module.
    
    Args:
        module: GraphModule to modify
        start: Node to preserve
        end: Node after which to insert
    """
    # Get the module from the start node
    preserve_module = get_node_from_module(module, start)
    # Remove the node from graph
    remove_node_from_module(module, start)
    # Add the preserved module after the end node
    add_module_after_node(module, start, end, preserve_module)
    # Lint and recompile
    lint_and_recompile(module)

def add_activation_to_node(model: GraphModule, node: Node, range_max: int) -> None:
    """Add activation quantization parameters to a node.
    
    Args:
        model: GraphModule containing the node
        node: Node to add activation to
        range_max: Maximum quantization range
    """
    module_attr = getattr(model, str(node))
    if hasattr(module_attr, "activation_post_process"):
        # Set quantization scale for residual inputs
        module_attr.activation_post_process.range_max = range_max
        module_attr.activation_post_process.fixed_range = True
        setattr(model, str(node), module_attr)

def set_quant_range(model: GraphModule, node: Node, target_name: str, quant_min: int, quant_max: int) -> None:
    """Set quantization range for a node.
    
    Args:
        model: GraphModule containing the node
        node: Node to set range for
        target_name: Target operation name
        quant_min: Minimum quantization value
        quant_max: Maximum quantization value
        
    Raises:
        IndexError: If node.args is empty
        RuntimeError: If setting range fails
    """
    if not node.args:
        raise IndexError("Node has no arguments to access")
    
    try:
        named_modules = dict(model.named_modules())
        module_attr = getattr(model, str(node), None)
        if module_attr is None:
            raise AttributeError(f"Model has no attribute for node '{str(node)}'")
        
        # Check if input is a ConvBn2d module for proper precision handling
        is_convbn_add = node.args[0].target in named_modules
        is_convbn_add = is_convbn_add and target_name == operator.add
        is_convbn_add = is_convbn_add and isinstance(named_modules[node.args[0].target], torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn2d)
        if is_convbn_add:
            # Set 10-bit signed range to prevent precision loss with 8-bit residual data
            if hasattr(module_attr, "quant_min") and hasattr(module_attr, "quant_max"):
                module_attr.quant_min = quant_min
                module_attr.quant_max = quant_max
                module_attr.dtype = torch.qint32
                setattr(model, str(node), module_attr)
    except Exception as e:
        raise RuntimeError(f"Failed to set quantization range: {str(e)}")

def adjust_residual_inputs_qconfig(model: GraphModule, range_max: int = 0, quant_min: int = torch.inf, quant_max: int = torch.inf) -> GraphModule:
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

def assign_same_observers(model: GraphModule, node_1: Node, node_2: Node) -> GraphModule:
    """Assign the same observers to two nodes' activation quantization.
    
    Args:
        model: GraphModule containing the nodes
        node_1: First node to get observer from
        node_2: Second node to assign observer to
        
    Returns:
        Modified GraphModule
    """
    activation_post_proc = None
    if hasattr(model, node_1.target):
        module_attr = getattr(model, node_1.target)
        if hasattr(module_attr, "activation_post_process"):
            activation_post_proc = module_attr.activation_post_process

    if activation_post_proc and hasattr(model, node_2.target):
        module_attr = getattr(model, node_2.target)
        setattr(module_attr, "activation_post_process", activation_post_proc)
        model.graph.lint()
        model.recompile()
        
    return model

def assign_same_observers_for_residual_inputs(model: GraphModule):
    """Assign same observers to all residual input node pairs.
    
    Finds nodes with residual operators and assigns matching observers
    to their input nodes.
    
    Args:
        model: GraphModule to modify
    """
    residual_operators = {operator.add, torch.add, "add", torch.cat, torch.stack}
    for node in model.graph.nodes:
        target_name = node.target
        if target_name in residual_operators:
            node_1, node_2 = node.args
            assign_same_observers(model, node_1, node_2)


def remove_identity(model: torch.nn.Module, verbose_mode: bool = False, **kwargs) -> torch.fx.GraphModule:
    """Remove `torch.nn.Identity` submodules from a traced model.

    The function traces the input `model` (if it is not already a
    `GraphModule`) and searches for `call_module` nodes referencing
    `torch.nn.Identity`. Matching identity nodes are replaced with their
    input value and the corresponding submodule is removed when it is not
    referenced elsewhere.

    Parameters
    - model: `nn.Module` or `GraphModule` to process.
    - verbose_mode: when True print progress and debug information.

    Returns
    - A `torch.fx.GraphModule` with identity nodes removed.
    """

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
                parent_name, name = _get_parent_name(node.target)
                modules[parent_name].__delattr__(name)
                modules.pop(node.target, None)
            traced_model.graph.erase_node(node)
            n += 1
        except Exception as e:
            if verbose_mode:
                print(n, e)

    lint_and_recompile(traced_model)
    return traced_model
