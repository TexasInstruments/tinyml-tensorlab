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

import copy
import torch
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping
from torch.autograd import grad
from logging import getLogger
from tabulate import tabulate

logger = getLogger("root.main.auto_quantization")


def compute_hessian_vector_product(model, param, v, inputs, targets, criterion):
    with torch.backends.cudnn.flags(enabled=False):
        outputs = model(inputs)
    loss = criterion(outputs, targets)
    grads = grad(loss, param, create_graph=True)[0]
    grad_v_product = torch.sum(grads * v)
    hvp = grad(grad_v_product, param, retain_graph=True)[0]
    return hvp


def power_iteration(model, param, inputs, targets, criterion, num_eigenvalues=1, num_iterations=100):
    eigenvalues = []
    param_shape = param.shape
    param_size = param.numel()
    for _ in range(num_eigenvalues):
        v = torch.randn(param_size, device=param.device)
        v = v / torch.norm(v)
        for _ in range(num_iterations):
            Hv = compute_hessian_vector_product(model, param, v.reshape(param_shape), inputs, targets, criterion)
            v_new = Hv.flatten()
            eigenvalue = torch.dot(v_new, v)
            v = v_new / torch.norm(v_new)
        eigenvalues.append(eigenvalue.item())
    return eigenvalues


def compute_hessian_eigenvalues(model, inputs, targets, criterion, num_eigenvalues=1, num_iterations=100):
    eigenvalues = {}
    was_training = model.training
    has_rnn = any(isinstance(m, torch.nn.RNNBase) for m in model.modules())
    if has_rnn:
        model.train()
    else:
        model.eval()
    inputs = inputs.detach().clone()
    targets = targets.detach().clone()

    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            eigenvalues[name] = power_iteration(model, param, inputs, targets, criterion, num_eigenvalues, num_iterations)
    if was_training:
        model.train()
    else:
        model.eval()
    return eigenvalues


def compute_hessian_sensitivity(model, inputs, targets, criterion,
                                num_eigenvalues=1, num_iterations=100, batch_size=128):
    """Computes per-layer hessian sensitivity"""
    if inputs is None or targets is None or criterion is None:
        logger.warning("Hessian sensitivity requires inputs, targets, and criterion.")
        return {}, {}
    eigenvalues = compute_hessian_eigenvalues(
        model, inputs[:batch_size], targets[:batch_size], criterion, num_eigenvalues, num_iterations
    )
    module_sensitivities = {}
    module_params = {}
    for name, eigs in eigenvalues.items():
        max_eig = max(abs(e) for e in eigs)
        module_name = name.rsplit('.', 1)[0]
        try:
            n_params = dict(model.named_parameters())[name].numel()
        except KeyError:
            n_params = 0
        if module_name in module_sensitivities:
            module_sensitivities[module_name] = max(module_sensitivities[module_name], max_eig)
            module_params[module_name] += n_params
        else:
            module_sensitivities[module_name] = max_eig
            module_params[module_name] = n_params
    if not module_sensitivities:
        logger.warning("No quantizable layers found for Hessian sensitivity analysis.")
    else:
        logger.info("Layer sensitivities (sorted by Hessian eigenvalue, descending):")
        for name, sens in sorted(module_sensitivities.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {name}: {sens:.6f} ({module_params[name]} params)")
    return module_sensitivities, module_params


def greedy_bit_allocation(module_sensitivities, module_params, target_avg_bitwidth):
    """Greedy algorithm for assigning bitwidths among {32, 8, 4, 2} proportional to sensitivity"""
    if not module_sensitivities:
        return {}
    precision_levels = [2, 4, 8, 32]
    total_params = sum(module_params.values())
    total_bit_budget = target_avg_bitwidth * total_params
    layer_bitwidths = {name: 2 for name in module_sensitivities}
    current_bits_used = sum(2 * module_params[l] for l in layer_bitwidths)
    while current_bits_used < total_bit_budget:
        best_layer = best_new_bitwidth = None
        best_ratio = -float('inf')
        for layer_name in layer_bitwidths:
            current_bitwidth = layer_bitwidths[layer_name]
            next_bitwidth = next((b for b in precision_levels if b > current_bitwidth), None)
            if next_bitwidth is None:
                continue
            n_params = module_params[layer_name]
            cost = (next_bitwidth - current_bitwidth) * n_params
            if current_bits_used + cost > total_bit_budget:
                continue
            reduction = module_sensitivities[layer_name] * n_params * (
                1.0 / (4 ** current_bitwidth) - 1.0 / (4 ** next_bitwidth)
            )
            ratio = reduction / cost if cost > 0 else 0
            if ratio > best_ratio:
                best_ratio, best_layer, best_new_bitwidth = ratio, layer_name, next_bitwidth
        if best_layer is None:
            break
        old_bw = layer_bitwidths[best_layer]
        layer_bitwidths[best_layer] = best_new_bitwidth
        current_bits_used += (best_new_bitwidth - old_bw) * module_params[best_layer]
    mixed_precision = {p: [] for p in precision_levels}
    for layer_name, bitwidth in layer_bitwidths.items():
        mixed_precision[bitwidth].append(layer_name)
    return {k: v for k, v in mixed_precision.items() if v}


def calibrate_and_evaluate(
    model: torch.nn.Module,
    qconfig_mapping,
    calibration_dataloader,
    eval_dataloader,
    task_type: str,
    example_inputs,
    device=None,
    num_calibration_batches: int = None,
) -> float:
    """Calibration for finding best avg. bitwidth to be assigned. Only a single evaluation step. Binary search algorithm stopping criteria"""
    model_copy = copy.deepcopy(model)
    try:
        device = next(model_copy.parameters()).device
    except StopIteration:
        device = torch.device(device) if device is not None else torch.device('cpu')
    model_copy = model_copy.to(device)
    model_copy.eval()
    prepared = quantize_fx.prepare_qat_fx(model_copy, qconfig_mapping, example_inputs)
    prepared.eval()
    with torch.no_grad():
        for batch_idx, (_, inputs, _) in enumerate(calibration_dataloader):
            if num_calibration_batches is not None and batch_idx >= num_calibration_batches:
                break
            if device is not None:
                inputs = inputs.to(device)
            prepared(inputs.float())
    prepared.eval()
    task_lower = task_type.lower()
    correct = 0
    total_samples = 0
    running_sum = 0.0
    total_elements = 0
    ss_res = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    reg_count = 0

    with torch.no_grad():
        for _, inputs, targets in eval_dataloader:
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)
            inputs_f = inputs.float()
            preds = prepared(inputs_f)

            if 'classification' in task_lower:
                correct += (preds.argmax(dim=1) == targets).sum().item()
                total_samples += targets.size(0)
            elif 'anomaly' in task_lower:
                running_sum += torch.sum((preds - inputs_f) ** 2).item()
                total_elements += inputs_f.numel()
            elif 'forecasting' in task_lower:
                p = preds.float().reshape(targets.shape)
                t = targets.float()
                running_sum += torch.sum(
                    2.0 * torch.abs(p - t) /
                    (torch.abs(p) + torch.abs(t) + 1e-8)
                ).item() * 100
                total_elements += t.numel()
            elif 'regression' in task_lower:
                t = targets.float().flatten()
                p = preds.float().flatten()
                ss_res += torch.sum((t - p) ** 2).item()
                sum_y += torch.sum(t).item()
                sum_y2 += torch.sum(t ** 2).item()
                reg_count += t.numel()

    if 'classification' in task_lower:
        return correct / total_samples if total_samples > 0 else float('nan')
    elif 'anomaly' in task_lower:
        return running_sum / total_elements if total_elements > 0 else float('nan')
    elif 'forecasting' in task_lower:
        return running_sum / total_elements if total_elements > 0 else float('nan')
    elif 'regression' in task_lower:
        ss_tot = sum_y2 - (sum_y ** 2) / reg_count if reg_count > 0 else 0.0
        return (1.0 - ss_res / ss_tot) if ss_tot != 0.0 else float('nan')
    else:
        logger.warning(f"calibrate_and_evaluate: unknown task_type '{task_type}', returning nan")
        return float('nan')


def find_optimal_bitwidth_binary_search(
    model: torch.nn.Module,
    calibration_dataloader,
    eval_dataloader,
    task_type: str,
    float_metric: float,
    example_inputs,
    module_sensitivities: dict,
    module_params: dict,
    qconfig_dict: dict = None,
    autoquant_tolerance_classification: float = 0.05,
    autoquant_tolerance_regression: float = 0.05,
    autoquant_tolerance_anomaly: float = 2,
    autoquant_tolerance_forecasting: float = 2.0,
    num_calibration_batches: int = None,
    device=None,
) -> float:
    """Binary search over target average bitwidths"""
    from . import qconfig_types
    task_lower = task_type.lower()
    if 'classification' in task_lower:
        range_lo, range_hi = 4, 8
        tolerance = autoquant_tolerance_classification
        higher_is_better = True
    elif 'anomaly' in task_lower:
        range_lo, range_hi = 4, 8
        tolerance = autoquant_tolerance_anomaly
        higher_is_better = False
    elif 'forecasting' in task_lower:
        range_lo, range_hi = 4, 32
        tolerance = autoquant_tolerance_forecasting
        higher_is_better = False
    else:
        range_lo, range_hi = 4, 12
        tolerance = autoquant_tolerance_regression
        higher_is_better = True

    if float_metric is None:
        logger.warning(
            "find_optimal_bitwidth_binary_search: float_metric is None, "
            f"defaulting to max target_avg_bitwidth {range_hi}"
        )
        return range_hi

    if higher_is_better:
        threshold = float_metric * (1.0 - tolerance)
    else:
        threshold = float_metric * (1.0 + tolerance)

    logger.info(
        f"Binary search bitwidth selection | task={task_type} | "
        f"float_metric={float_metric:.4f} | threshold={threshold:.4f} | "
        f"higher_is_better={higher_is_better} | search range=[{range_lo}, {range_hi}]"
    )

    base = qconfig_dict or {}
    best_bitwidth = range_hi
    lo, hi = range_lo, range_hi

    while lo <= hi:
        mid = (lo + hi) // 2
        mixed_precision = greedy_bit_allocation(module_sensitivities, module_params, mid)
        base_bw = min(mid, 8)
        bw_qconfig_dict = {
            'weight': {**base.get('weight', {}), 'bitwidth': base_bw},
            'activation': {**base.get('activation', {}), 'bitwidth': base_bw},
        }
        probe_mapping = QConfigMapping().set_global(qconfig_types.get_default_qconfig(bw_qconfig_dict))
        qcd_copy = {'weight': dict(base.get('weight', {})), 'activation': dict(base.get('activation', {}))}
        probe_mapping = qconfig_types.apply_mixed_precision(probe_mapping, qcd_copy, mixed_precision)
        metric = calibrate_and_evaluate(
            model=model,
            qconfig_mapping=probe_mapping,
            calibration_dataloader=calibration_dataloader,
            eval_dataloader=eval_dataloader,
            task_type=task_type,
            example_inputs=example_inputs,
            device=device,
            num_calibration_batches=num_calibration_batches,
        )
        if metric != metric:
            passes = False
        elif higher_is_better:
            passes = metric >= threshold
        else:
            passes = metric <= threshold
        logger.info(
            f"  target_avg_bitwidth={mid} | metric={metric:.4f} | "
            f"threshold={threshold:.4f} | {'PASS' if passes else 'FAIL'}"
        )
        if passes:
            best_bitwidth = mid
            hi = mid - 1
        else:
            lo = mid + 1
    if best_bitwidth == range_hi:
        logger.warning(
            f"find_optimal_bitwidth_binary_search: no target_avg below {range_hi} "
            f"passed the threshold; using {range_hi}"
        )
    logger.info(f"Binary search complete | selected target_avg_bitwidth={best_bitwidth}")
    return best_bitwidth


def run_auto_quantization(model, qconfig_dict, qconfig_mapping, get_default_qconfig_fn, apply_mixed_precision_fn):
    logger.warning(
        "auto_quantization=True: quantization_weight_bitwidth and quantization_activation_bitwidth "
        "will be overridden by Hessian-based auto quantization. "
        "Set auto_quantization=False in the config.yaml file for uniform specified bitwidths."
    )
    inputs = qconfig_dict.get('inputs')
    targets = qconfig_dict.get('targets')
    criterion = qconfig_dict.get('criterion')
    module_sensitivities, module_params = compute_hessian_sensitivity(model, inputs, targets, criterion)

    optimal_bitwidth = None
    if qconfig_dict.get('calibration_dataloader') is not None and module_sensitivities:
        bsearch_kwargs = dict(
            model=model,
            calibration_dataloader=qconfig_dict.get('calibration_dataloader'),
            eval_dataloader=qconfig_dict.get('eval_dataloader'),
            task_type=qconfig_dict.get('task_type', 'classification'),
            float_metric=qconfig_dict.get('float_metric'),
            example_inputs=qconfig_dict.get('example_inputs'),
            module_sensitivities=module_sensitivities,
            module_params=module_params,
            qconfig_dict=qconfig_dict,
            device=qconfig_dict.get('device'),
        )
        for tolerance_key in (
            'autoquant_tolerance_classification',
            'autoquant_tolerance_regression',
            'autoquant_tolerance_anomaly',
            'autoquant_tolerance_forecasting',
        ):
            if qconfig_dict.get(tolerance_key) is not None:
                bsearch_kwargs[tolerance_key] = qconfig_dict[tolerance_key]
        optimal_bitwidth = find_optimal_bitwidth_binary_search(**bsearch_kwargs)

    total_params = sum(module_params.values())
    total_bit_budget = optimal_bitwidth * total_params
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Target average bitwidth: {optimal_bitwidth}")
    logger.info(f"Total bit budget: {total_bit_budget}")
    logger.info("Starting greedy bit allocation...")
    mixed_precision = greedy_bit_allocation(module_sensitivities, module_params, optimal_bitwidth)
    actual_bits = sum(bw * sum(module_params[l] for l in layers) for bw, layers in mixed_precision.items())
    actual_avg = actual_bits / total_params if total_params > 0 else 0
    logger.info(f"Bit allocation complete. Actual average bitwidth: {actual_avg:.2f}")
    logger.info(f"Bits used: {actual_bits} / {total_bit_budget}")
    rows = []
    for p in sorted(mixed_precision.keys()):
        for layer_name in mixed_precision[p]:
            try:
                layer_type = type(model.get_submodule(layer_name)).__name__
            except AttributeError:
                layer_type = "unknown"
            params = module_params[layer_name]
            pct = 100.0 * params / total_params if total_params > 0 else 0.0
            rows.append([f"{layer_name} ({layer_type})", p, params, f"{pct:.2f}%"])
    logger.info("Hessian-based precision assignment (greedy bit budget):\n{}".format(
        tabulate(rows, headers=["Layer (Type)", "Assigned bitwidth", "Params", "% of Total Params"], tablefmt="grid")))
    final_bw = min(int(optimal_bitwidth), 8)
    bw_qconfig_dict = {
        'weight': {**qconfig_dict.get('weight', {}), 'bitwidth': final_bw},
        'activation': {**qconfig_dict.get('activation', {}), 'bitwidth': final_bw},
    }
    if any(bw < 32 for bw in mixed_precision.keys()):
        qconfig_mapping = QConfigMapping().set_global(get_default_qconfig_fn(bw_qconfig_dict))
        qconfig_mapping = apply_mixed_precision_fn(qconfig_mapping, qconfig_dict, mixed_precision)
    else:
        logger.warning(
            "All layers assigned 32-bit by binary search — no compression found. "
            "Disabling quantization entirely; model will remain float32."
        )
        qconfig_mapping = QConfigMapping()
    return qconfig_mapping