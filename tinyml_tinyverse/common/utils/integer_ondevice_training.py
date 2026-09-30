"""
integer_ondevice_training.py
----------------------------
Generates C artifacts for integer on-device learning (ODL) from a QAT ONNX model.

Public API
----------
export_training_data(dataset_train, dataset_val, dataset_test, args, onnx_path=None)
    Sample balanced ODL training data, run output distribution analysis, and write
    train_data.h / train_data.c. Returns (target_mags_correct, target_mags_wrong).

export_for_ondevice_training(onnx_path, args, target_mags_correct, target_mags_wrong)
    Parse QAT ONNX and write integer_model_config.h / integer_model_config.c.

export_frozen_model(onnx_path, args)
    Split QAT ONNX at the trainable boundary and save the frozen prefix for TVM.

Output directory: <args.output_dir>/integer_model_config/
"""

import ast
import logging
import os
from math import ceil

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
import onnxruntime as ort
from tinyml_tinyverse.common.utils.ondevice_training import extract_frozen_subgraph

logger = logging.getLogger(__name__)

COMPUTE_OPS = {'Conv', 'MatMul', 'Gemm'}
SKIP_OPS    = {'Reshape'}

SUPPORTED_BLOCKS = [
    ('PREQUANT',   ['Add', 'Mul', 'Mul', 'Floor', 'Clip'],                            False),
    ('CONV2DRELU', ['Conv', 'Add', 'Mul', 'Mul', 'Floor', 'Clip', 'Relu', 'Clip'],    True),
    ('CONV2D',     ['Conv', 'Add', 'Mul', 'Mul', 'Floor', 'Clip'],                    True),
    ('LINEARRELU', ['MatMul', 'Add', 'Mul', 'Mul', 'Floor', 'Clip', 'Relu', 'Clip'],  True),
    ('LINEAR',     ['MatMul', 'Add', 'Mul', 'Mul', 'Floor', 'Clip'],                  True),
    ('GEMMRELU',   ['Gemm', 'Mul', 'Mul', 'Floor', 'Clip', 'Relu', 'Clip'],           True),
    ('GEMM',       ['Gemm', 'Mul', 'Mul', 'Floor', 'Clip'],                           True),
    ('RESHAPE',    ['Reshape'],                                                        False),
]
SUPPORTED_BLOCKS.sort(key=lambda x: len(x[1]), reverse=True)

_RELU_BLOCKS = {'CONV2DRELU', 'LINEARRELU', 'GEMMRELU'}

BLOCK_TO_CTYPE = {
    'PREQUANT':   'INT_LAYER_PREQUANT',
    'CONV2D':     'INT_LAYER_CONV2D',
    'CONV2DRELU': 'INT_LAYER_CONV2D_RELU',
    'LINEAR':     'INT_LAYER_LINEAR',
    'LINEARRELU': 'INT_LAYER_LINEAR_RELU',
    'GEMM':       'INT_LAYER_LINEAR',
    'GEMMRELU':   'INT_LAYER_LINEAR_RELU',
}

# ODL training defaults. Can be overridden in C main.c.
TARGET_MAG_DEFAULT = 64   
MU_WEIGHT_DEFAULT  = 1
MU_OFFSET_DEFAULT  = 1


def load_graph(onnx_path):
    """Load an ONNX model, run shape inference, validate it, and return a topologically
    sorted onnx-graphsurgeon Graph."""
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model, data_prop=True)
    onnx.checker.check_model(model)
    graph = gs.import_onnx(model)
    graph.toposort()
    return graph


def segment_graph(graph):
    """Split a graph into per-block node lists.

    The first segment (if present) is a pre-compute preamble (e.g. PREQUANT ops).
    Each subsequent segment starts with a COMPUTE_OP (Conv / MatMul / Gemm) and
    contains all following non-compute, non-skip nodes up to the next COMPUTE_OP.
    SKIP_OPS (Reshape) each form their own single-node segment.

    Args:
        graph: onnx-graphsurgeon Graph (topologically sorted).

    Returns:
        list of lists of onnx-graphsurgeon Node objects.
    """
    nodes = [n for n in graph.nodes if n.op != 'Constant']
    segments = []
    i = 0
    pre = []
    while i < len(nodes) and nodes[i].op not in COMPUTE_OPS | SKIP_OPS:
        pre.append(nodes[i])
        i += 1
    if pre:
        segments.append(pre)
    while i < len(nodes):
        if nodes[i].op in SKIP_OPS:
            segments.append([nodes[i]])
            i += 1
        elif nodes[i].op in COMPUTE_OPS:
            seg = [nodes[i]]
            i += 1
            while i < len(nodes) and nodes[i].op not in COMPUTE_OPS | SKIP_OPS:
                seg.append(nodes[i])
                i += 1
            segments.append(seg)
        else:
            raise RuntimeError(
                f"Unsupported op '{nodes[i].op}' at position {i}. Expected Conv, MatMul, Gemm, or Reshape."
            )
    return segments


def classify_segments(segments):
    """Match each segment to a known block type from SUPPORTED_BLOCKS.

    Args:
        segments: output of segment_graph().

    Returns:
        list of (block_name, creates_params, seg) tuples.

    Raises:
        RuntimeError if a segment does not match any known op pattern.
    """
    blocks = []
    for seg in segments:
        ops = [n.op for n in seg]
        matched = False
        for name, pattern, creates in SUPPORTED_BLOCKS:
            if ops == pattern:
                blocks.append((name, creates, seg))
                matched = True
                break
        if not matched:
            raise RuntimeError(
                f"Unrecognized op sequence: {ops}\n"
                f"Supported: {[b[0] for b in SUPPORTED_BLOCKS]}"
            )
    return blocks


def _get_clip_bounds(clip_node):
    """Return (lo, hi) float clip bounds from an ONNX Clip node.

    onnx-graphsurgeon folds Constant inputs into gs.Constant objects, so
    inputs[1].values and inputs[2].values are directly accessible as scalars.
    """
    lo = float(clip_node.inputs[1].values)
    hi = float(clip_node.inputs[2].values)
    return lo, hi


def extract_block(block_name, seg):
    """Extract weights, quantization parameters, and shape metadata from one block.

    Args:
        block_name: str, one of SUPPORTED_BLOCKS names (e.g. 'PREQUANT', 'CONV2DRELU').
        seg: list of onnx-graphsurgeon Node objects for this block.

    Returns:
        dict with keys: block_type, in_size, out_size, clip_lo, clip_hi, and
        block-specific keys (weights, offset_arr, mult_arr, k_arr, n_filters, etc.).
    """
    info = {'block_type': block_name}

    if block_name == 'PREQUANT':
        add_node, mul1_node, mul2_node, floor_node, clip_node = seg
        offset_arr = add_node.inputs[1].values.flatten()
        scale_raw  = mul1_node.inputs[1].values.flatten()
        shift_raw  = mul2_node.inputs[1].values.flatten()
        scale_arr  = scale_raw * shift_raw
        in_shape   = list(add_node.inputs[0].shape)
        in_size    = int(np.prod(in_shape[1:]))
        clip_lo, clip_hi = _get_clip_bounds(clip_node)
        info.update({
            'in_size':    in_size,
            'out_size':   in_size,
            'n_ch':       len(offset_arr),
            'offset_arr': offset_arr.astype(np.float32),
            'scale_arr':  scale_arr.astype(np.float32),
            'clip_lo':    clip_lo,
            'clip_hi':    clip_hi,
        })

    elif block_name in ('CONV2DRELU', 'CONV2D'):
        conv_node = seg[0]
        add_node  = seg[1]
        mul1_node = seg[2]
        mul2_node = seg[3]
        last_clip = seg[-1]

        strides   = conv_node.attrs.get('strides',   [1, 1])
        pads      = conv_node.attrs.get('pads',      [0, 0, 0, 0])
        dilations = conv_node.attrs.get('dilations', [1, 1])
        group     = conv_node.attrs.get('group',     1)
        if any(p != 0 for p in pads):
            if pads[0] != pads[2]:
                raise ValueError(
                    f"Conv '{conv_node.name}': asymmetric H padding "
                    f"[top={pads[0]}, bottom={pads[2]}]. Only symmetric padding supported."
                )
            if pads[1] != pads[3]:
                raise ValueError(
                    f"Conv '{conv_node.name}': asymmetric W padding "
                    f"[left={pads[1]}, right={pads[3]}]. Only symmetric padding supported."
                )
        stride_h, stride_w = int(strides[0]), int(strides[1])
        pad_h, pad_w = int(pads[0]), int(pads[1])
        if any(d != 1 for d in dilations):
            raise ValueError(
                f"Conv '{conv_node.name}': dilations={dilations}. Only dilation=1 is supported."
            )
        if group != 1:
            raise ValueError(
                f"Conv '{conv_node.name}': group={group}. Only group=1 (standard conv) is supported."
            )

        weights_raw = conv_node.inputs[1].values
        weights = np.round(weights_raw).astype(np.int8)
        n_filters, _, kH, kW = weights.shape
        offset_arr = add_node.inputs[1].values.flatten()
        mult_arr   = mul1_node.inputs[1].values.flatten()
        shift_arr  = mul2_node.inputs[1].values.flatten()
        k_arr      = np.round(-np.log2(np.abs(shift_arr))).astype(np.int8)
        clip_lo, clip_hi = _get_clip_bounds(last_clip)
        _, _, conv_out_h, conv_out_w = conv_node.outputs[0].shape
        out_size = int(n_filters * conv_out_h * conv_out_w)
        in_size  = int(np.prod(list(conv_node.inputs[0].shape)[1:]))
        info.update({
            'in_size':     in_size,
            'out_size':    out_size,
            'weights':     weights,
            'offset_arr':  np.round(offset_arr).astype(np.int32),
            'mult_arr':    np.round(mult_arr).astype(np.int8),
            'k_arr':       k_arr,
            'clip_lo':     clip_lo,
            'clip_hi':     clip_hi,
            'n_filters':   int(n_filters),
            'kH':          int(kH),
            'kW':          int(kW),
            'conv_out_h':  int(conv_out_h),
            'conv_out_w':  int(conv_out_w),
            'stride_h':    stride_h,
            'stride_w':    stride_w,
            'pad_h':       pad_h,
            'pad_w':       pad_w,
            'input_h':     int(conv_node.inputs[0].shape[2]),
            'input_w':     int(conv_node.inputs[0].shape[3]),
        })

    elif block_name in ('LINEARRELU', 'LINEAR', 'GEMMRELU', 'GEMM'):
        mm_node   = seg[0]
        has_add   = (seg[1].op == 'Add')
        mul1_node = seg[2] if has_add else seg[1]
        mul2_node = seg[3] if has_add else seg[2]
        last_clip = seg[-1]

        weights_raw = mm_node.inputs[1].values
        weights = np.round(weights_raw).astype(np.int8)
        in_size, out_size = weights.shape

        if has_add:
            offset_arr = seg[1].inputs[1].values.flatten()
        else:
            offset_arr = mm_node.inputs[2].values.flatten()
        mult_arr  = mul1_node.inputs[1].values.flatten()
        shift_arr = mul2_node.inputs[1].values.flatten()
        k_arr     = np.round(-np.log2(np.abs(shift_arr))).astype(np.int8)
        clip_lo, clip_hi = _get_clip_bounds(last_clip)
        info.update({
            'in_size':    int(in_size),
            'out_size':   int(out_size),
            'weights':    weights,
            'offset_arr': np.round(offset_arr).astype(np.int32),
            'mult_arr':   np.round(mult_arr).astype(np.int8),
            'k_arr':      k_arr,
            'clip_lo':    clip_lo,
            'clip_hi':    clip_hi,
        })

    return info


def extract_all_blocks(blocks):
    """Extract parameters from all non-RESHAPE blocks.

    Args:
        blocks: output of classify_segments().

    Returns:
        list of layer dicts (one per non-RESHAPE block), in graph order.
    """
    extracted = []
    for name, creates, seg in blocks:
        if name == 'RESHAPE':
            continue
        extracted.append(extract_block(name, seg))
    return extracted


def _fmt_array_i8(arr, per_row=16):
    """Format a numpy array as a C int8 initializer body (no braces), 16 values per row."""
    vals = arr.flatten().tolist()
    rows = [vals[i:i+per_row] for i in range(0, len(vals), per_row)]
    return '    ' + ',\n    '.join(', '.join(f'{v:4d}' for v in row) for row in rows)


def _fmt_array_i32(arr, per_row=8):
    """Format a numpy array as a C int32 initializer body (no braces), 8 values per row."""
    vals = arr.flatten().tolist()
    rows = [vals[i:i+per_row] for i in range(0, len(vals), per_row)]
    return '    ' + ',\n    '.join(', '.join(f'{v:8d}' for v in row) for row in rows)


def _fmt_f32(v):
    """Format a scalar float as a C float literal with 'f' suffix (e.g. 8.0f, 7.52f)."""
    s = f'{v:.8g}'
    if '.' not in s and 'e' not in s and 'E' not in s:
        s += '.0'
    return s + 'f'


def _fmt_array_f32(arr, per_row=8):
    """Format a numpy array as a C float initializer body (no braces), 8 values per row."""
    vals = arr.flatten().tolist()
    rows = [vals[i:i+per_row] for i in range(0, len(vals), per_row)]
    return '    ' + ',\n    '.join(', '.join(_fmt_f32(v) for v in row) for row in rows)


def emit_integer_model_config_h(layers, meta, out_dir,
                                target_mags_correct=None, target_mags_wrong=None,
                                is_full_model=False):
    """Write integer_model_config.h to out_dir.

    Emits model dimensions (INPUT_SIZE, N_CLASSES), ODL training defaults
    (BATCH_SIZE, TARGET_MAG_DEFAULT, MU_WEIGHT_DEFAULT, MU_OFFSET_DEFAULT),
    per-class target arrays (TARGET_MAG_CORRECT, TARGET_MAG_WRONG), and
    extern declarations for INT_LAYERS and INT_MODEL.

    Args:
        layers              : list of layer dicts from extract_all_blocks().
        meta                : dict with keys 'input_size', 'n_classes', 'n_trainable'.
        out_dir             : output directory path.
        target_mags_correct : list[int] of length n_classes, or None to use defaults.
        target_mags_wrong   : list[int] of length n_classes (negative), or None to use defaults.
        is_full_model       : bool, True if all compute layers are trainable (PREQUANT-only TVM frozen model).
    """
    n_layers    = len(layers)
    n_trainable = meta['n_trainable']
    input_size  = meta['input_size']
    n_classes   = meta['n_classes']

    def _pad(lst, default, n):
        lst = list(lst) if lst is not None else []
        while len(lst) < n:
            lst.append(default)
        return lst[:n]

    target_mags_correct = _pad(target_mags_correct,  TARGET_MAG_DEFAULT,  n_classes)
    target_mags_wrong   = _pad(target_mags_wrong,    -TARGET_MAG_DEFAULT, n_classes)

    path = os.path.join(out_dir, 'integer_model_config.h')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('#ifndef INTEGER_MODEL_CONFIG_H\n')
        f.write('#define INTEGER_MODEL_CONFIG_H\n\n')
        f.write('#include "integer_odl.h"\n\n')

        f.write('// 0 = partial model\n')
        f.write('// 1 = full model\n')
        f.write(f'#define IS_FULL_MODEL_RETRAIN  {1 if is_full_model else 0}\n\n')

        f.write('// Model dimensions\n')
        f.write(f'#define INPUT_SIZE          {input_size}\n')
        frozen_out = layers[0]['in_size']
        f.write(f'#define FROZEN_OUTPUT_SIZE  {frozen_out}\n')
        f.write(f'#define N_CLASSES           {n_classes}\n\n')
        f.write(f'#define BATCH_SIZE          8\n')
        correct_vals = ', '.join(f'{v:4d}' for v in target_mags_correct)
        wrong_vals   = ', '.join(f'{v:4d}' for v in target_mags_wrong)
        f.write(f'// Mean correct-class logit per class\n')
        f.write(f'static const int8_t TARGET_MAG_CORRECT[N_CLASSES] = {{ {correct_vals} }};\n')
        f.write(f'// Mean wrong-class logit per class\n')
        f.write(f'static const int8_t TARGET_MAG_WRONG[N_CLASSES]   = {{ {wrong_vals} }};\n')
        f.write(f'#define MU_WEIGHT_DEFAULT   {MU_WEIGHT_DEFAULT}\n')
        f.write(f'#define MU_OFFSET_DEFAULT   {MU_OFFSET_DEFAULT}\n\n')

        f.write('// Model descriptor\n')
        f.write(f'#define N_INT_LAYERS        {n_layers}\n')
        f.write(f'#define N_TRAINABLE_LAYERS  {n_trainable}\n\n')

        f.write('extern IntLayerDesc_t INT_LAYERS[N_INT_LAYERS];\n')
        f.write('extern IntModelCtx_t  INT_MODEL;\n\n')

        f.write('#endif // INTEGER_MODEL_CONFIG_H\n')

    logger.info(f'integer_model_config.h {path}')


def emit_integer_model_config_c(layers, meta, out_dir, is_full_model=False):
    """Write integer_model_config.c to out_dir.

    Emits forward declarations for all layer buffers, the INT_LAYERS descriptor
    array, the INT_MODEL context struct, and all weight / parameter arrays.

    grad_buf is sized to cover both the largest activation buffer and the first
    layer input size (needed because backward writes dx of layers[0].in_size into
    grad_buf for the first layer).

    Args:
        layers        : list of layer dicts from extract_all_blocks(). Never contains PREQUANT.
        meta          : dict with keys 'input_size', 'n_classes', 'n_trainable'.
        out_dir       : output directory path.
        is_full_model : True for full model ODL (input_type INT_INPUT_INT8),
                        False for partial model (input_type INT_INPUT_UINT8).
    """
    max_out = max(l['out_size'] for l in layers)
    max_buf = max(max_out, layers[0]['in_size'])

    path = os.path.join(out_dir, 'integer_model_config.c')
    with open(path, 'w', encoding='utf-8') as f:
        # topology comment
        f.write('// Model topology\n')
        for i, l in enumerate(layers):
            bt = l['block_type']
            extra = (f'  ({l["n_filters"]} filters, {l["kH"]}x{l["kW"]} kernel)'
                     if bt in ('CONV2D', 'CONV2DRELU') else '')
            f.write(f'//   layer{i}: {bt:<14} [{l["in_size"]:>4} -> {l["out_size"]:>4}]{extra}\n')
        f.write('\n')
        f.write('#include "integer_model_config.h"\n\n')

        def _fd(typ, name):
            return f'static {typ:<14} {name};\n'

        f.write('// Forward declarations\n')
        for i, l in enumerate(layers):
            bt = l['block_type']
            f.write(_fd('int8_t',       f'layer{i}_weights[]'))
            f.write(_fd('int32_t',      f'layer{i}_offset[]'))
            f.write(_fd('const int8_t', f'layer{i}_mult[]'))
            f.write(_fd('const int8_t', f'layer{i}_k[]'))
            act_type = 'uint8_t' if bt in _RELU_BLOCKS else 'int8_t'
            f.write(_fd(act_type, f'layer{i}_act_buf[BATCH_SIZE * {l["out_size"]}]'))
            if bt in _RELU_BLOCKS:
                mask_bytes = ceil(l['out_size'] / 8)
                f.write(_fd('uint8_t', f'layer{i}_relu_mask[BATCH_SIZE * {mask_bytes}]'))
        f.write(_fd('int8_t', f'grad_buf_0[BATCH_SIZE * {max_buf}]'))
        f.write(_fd('int8_t', f'grad_buf_1[BATCH_SIZE * {max_buf}]'))
        f.write('\n')

        f.write('// Layer descriptors\n')
        f.write('IntLayerDesc_t INT_LAYERS[N_INT_LAYERS] = {\n')
        for i, l in enumerate(layers):
            bt    = l['block_type']
            ctype = BLOCK_TO_CTYPE[bt]
            # clip_lo/clip_hi are int32_t in the C struct; emit as integer literals
            clip_lo = int(round(l['clip_lo']))
            clip_hi = int(round(l['clip_hi']))
            f.write(f'    {{\n')
            f.write(f'        .type     = {ctype},\n')
            f.write(f'        .in_size  = {l["in_size"]},\n')
            f.write(f'        .out_size = {l["out_size"]},\n')
            f.write(f'        .clip_lo  = {clip_lo},\n')
            f.write(f'        .clip_hi  = {clip_hi},\n')
            f.write(f'        .act_buf  = layer{i}_act_buf,\n')
            if bt in _RELU_BLOCKS:
                f.write(f'        .relu_mask = layer{i}_relu_mask,\n')
            else:
                f.write(f'        .relu_mask = NULL,\n')
            if bt in ('CONV2D', 'CONV2DRELU'):
                f.write(f'        .conv = {{\n')
                f.write(f'            .weights    = layer{i}_weights,\n')
                f.write(f'            .offset     = layer{i}_offset,\n')
                f.write(f'            .mult       = layer{i}_mult,\n')
                f.write(f'            .k          = layer{i}_k,\n')
                f.write(f'            .n_filters  = {l["n_filters"]},\n')
                f.write(f'            .kH         = {l["kH"]},\n')
                f.write(f'            .kW         = {l["kW"]},\n')
                f.write(f'            .conv_out_h = {l["conv_out_h"]},\n')
                f.write(f'            .conv_out_w = {l["conv_out_w"]},\n')
                f.write(f'            .stride_h   = {l.get("stride_h", 1)},\n')
                f.write(f'            .stride_w   = {l.get("stride_w", 1)},\n')
                f.write(f'            .pad_h      = {l.get("pad_h", 0)},\n')
                f.write(f'            .pad_w      = {l.get("pad_w", 0)},\n')
                f.write(f'            .input_h    = {l.get("input_h", 0)},\n')
                f.write(f'            .input_w    = {l.get("input_w", 0)},\n')
                f.write(f'        }},\n')
            else:  # LINEAR / LINEARRELU / GEMM / GEMMRELU
                f.write(f'        .linear = {{\n')
                f.write(f'            .weights = layer{i}_weights,\n')
                f.write(f'            .offset  = layer{i}_offset,\n')
                f.write(f'            .mult    = layer{i}_mult,\n')
                f.write(f'            .k       = layer{i}_k,\n')
                f.write(f'        }},\n')
            f.write(f'    }},\n')
        f.write('};\n\n')

        input_type = 'INT_INPUT_INT8' if is_full_model else 'INT_INPUT_UINT8'
        f.write('IntModelCtx_t INT_MODEL = {\n')
        f.write('    .n_layers        = N_INT_LAYERS,\n')
        f.write('    .layers          = INT_LAYERS,\n')
        f.write('    .grad_buf        = { grad_buf_0, grad_buf_1 },\n')
        f.write('    .mu_weight       = MU_WEIGHT_DEFAULT,\n')
        f.write('    .mu_offset       = MU_OFFSET_DEFAULT,\n')
        f.write(f'    .input_type      = {input_type},\n')
        f.write('};\n\n')

        f.write('// Parameter arrays\n')
        for i, l in enumerate(layers):
            bt = l['block_type']
            f.write(f'// layer{i}: {bt}\n')
            f.write(f'static int8_t layer{i}_weights[] = {{\n')
            f.write(_fmt_array_i8(l['weights']))
            f.write('\n};\n')
            f.write(f'static int32_t layer{i}_offset[] = {{\n')
            f.write(_fmt_array_i32(l['offset_arr']))
            f.write('\n};\n')
            f.write(f'static const int8_t layer{i}_mult[] = {{\n')
            f.write(_fmt_array_i8(l['mult_arr']))
            f.write('\n};\n')
            f.write(f'static const int8_t layer{i}_k[] = {{\n')
            f.write(_fmt_array_i8(l['k_arr']))
            f.write('\n};\n\n')

    logger.info(f'integer_model_config.c generated at {path}')


def emit_train_data_h(n_train, n_val, n_test, out_dir):
    """Write train_data.h to out_dir.

    Declares TRAIN_INPUTS, TRAIN_LABELS, VAL_INPUTS, VAL_LABELS, TEST_INPUTS,
    TEST_LABELS arrays and the corresponding sample-count defines.

    Args:
        n_train : number of training samples.
        n_val   : number of validation samples.
        n_test  : number of test samples.
        out_dir : output directory path.
    """
    path = os.path.join(out_dir, 'train_data.h')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('#ifndef TRAIN_DATA_H\n')
        f.write('#define TRAIN_DATA_H\n\n')
        f.write('#include <stdint.h>\n')
        f.write('#include "integer_model_config.h"\n\n')
        f.write(f'#define N_TRAIN_SAMPLES    {n_train}\n')
        f.write(f'#define N_VAL_SAMPLES      {n_val}\n')
        f.write(f'#define N_TEST_SAMPLES     {n_test}\n\n')
        f.write('extern const float  TRAIN_INPUTS [N_TRAIN_SAMPLES][INPUT_SIZE];\n')
        f.write('extern const int8_t TRAIN_LABELS [N_TRAIN_SAMPLES];\n')
        f.write('extern const float  VAL_INPUTS   [N_VAL_SAMPLES  ][INPUT_SIZE];\n')
        f.write('extern const int8_t VAL_LABELS   [N_VAL_SAMPLES  ];\n')
        f.write('extern const float  TEST_INPUTS  [N_TEST_SAMPLES ][INPUT_SIZE];\n')
        f.write('extern const int8_t TEST_LABELS  [N_TEST_SAMPLES ];\n\n')
        f.write('#endif // TRAIN_DATA_H\n')
    logger.info(f'integer_model_config.h generated at {path}')


def _emit_dataset(f, prefix, inputs, labels):
    """Write one dataset (inputs + labels) as C arrays to an open file.

    Args:
        f      : open file object.
        prefix : array name prefix, one of TRAIN, VAL, TEST.
        inputs : np.ndarray [N, input_size] float32.
        labels : np.ndarray [N] int8.
    """
    n = len(inputs)
    f.write(f'const float {prefix}_INPUTS[N_{prefix}_SAMPLES][INPUT_SIZE] = {{\n')
    for idx, (row, lbl) in enumerate(zip(inputs, labels)):
        vals = ', '.join(_fmt_f32(v) for v in row.flatten())
        f.write(f'    /* [{idx:4d}] label={int(lbl):2d} */ {{ {vals} }},\n')
    f.write('};\n\n')

    f.write(f'const int8_t {prefix}_LABELS[N_{prefix}_SAMPLES] = {{\n    ')
    f.write(', '.join(str(int(v)) for v in labels))
    f.write('\n};\n\n')


def emit_train_data_c(train_data, val_data, test_data, out_dir):
    """Write train_data.c to out_dir.

    Args:
        train_data : (X_train, Y_train) tuple of np.ndarrays.
        val_data   : (X_val, Y_val) tuple of np.ndarrays.
        test_data  : (X_test, Y_test) tuple of np.ndarrays.
        out_dir    : output directory path.
    """
    (X_train, Y_train) = train_data
    (X_val,   Y_val)   = val_data
    (X_test,  Y_test)  = test_data

    path = os.path.join(out_dir, 'train_data.c')
    with open(path, 'w', encoding='utf-8') as f:
        n_total = len(X_train) + len(X_val) + len(X_test)
        f.write(f'// {n_total} total samples (train={len(X_train)}, val={len(X_val)}, test={len(X_test)})\n')
        f.write('#include "train_data.h"\n\n')
        _emit_dataset(f, 'TRAIN', X_train, Y_train)
        _emit_dataset(f, 'VAL',   X_val,   Y_val)
        _emit_dataset(f, 'TEST',  X_test,  Y_test)

    logger.info(f'integer_model_config.c generated at {path}')


def _collect_samples(dataset, n_per_class):
    """Sample up to n_per_class examples per class from a dataset.

    Samples are selected randomly without replacement. If fewer than n_per_class
    examples are available for a class, all available examples are used and a
    warning is logged.

    Args:
        dataset     : GenericTSDataset
        n_per_class : number of samples to draw per class.

    Returns:
        inputs : np.ndarray [N, input_size] float32.
        labels : np.ndarray [N] int8.

    Raises:
        ValueError if no samples are found across all classes.
    """
    n_classes = len(dataset.classes)
    Y = np.array(dataset.Y)
    X = np.array(dataset.X)

    inputs_out, labels_out = [], []
    for cls_idx in range(n_classes):
        indices = np.where(Y == cls_idx)[0]
        n_avail = len(indices)
        if n_avail == 0:
            logger.warning(f'class {cls_idx}: no samples found')
            continue
        n_take = min(n_per_class, n_avail)
        if n_take < n_per_class:
            logger.warning(
                f'class {cls_idx}: only {n_avail} available, requested {n_per_class}'
            )
        chosen = np.random.choice(indices, size=n_take, replace=False)
        for idx in chosen:
            inputs_out.append(X[idx].flatten().astype(np.float32))
            labels_out.append(cls_idx)

    if not inputs_out:
        raise ValueError('No samples collected. Check that dataset.X and dataset.Y are populated.')

    return (np.array(inputs_out, dtype=np.float32), np.array(labels_out, dtype=np.int8))


def analyze_output_distribution(onnx_path, X_samples, Y_samples, class_names):
    """Run the ONNX on training samples and derive per-class TARGET_MAG(Target magnitude) values.

    TARGET_MAG_CORRECT[c] = mean of logit[:,c] where label == c.
    TARGET_MAG_WRONG[c] = mean of logit[:,c] where label != c.
    Prints and logs per-class statistics.

    Args:
        onnx_path   : path to QAT model.onnx.
        X_samples   : np.ndarray [N, input_size] float32, ODL training samples.
        Y_samples   : np.ndarray [N] int, true class indices (0 to n_classes-1).
        class_names : list[str] of class names.

    Returns:
        (target_mags_correct, target_mags_wrong) : tuple of two list[int], each of
        length n_classes. Falls back to defaults if onnxruntime is unavailable.
    """
    n_classes   = len(class_names)
    sess       = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inp        = sess.get_inputs()[0]
    input_name = inp.name
    inp_shape  = inp.shape          
    flat_size  = X_samples.shape[1]

    # Build concrete per-sample input shape (batch=1).
    # Keep all known positive-int spatial dims; if they multiply to flat_size use
    # them directly, otherwise fall back to [1, flat_size].
    static_spatial = [int(d) for d in inp_shape[1:] if isinstance(d, int) and d > 0]
    if static_spatial and int(np.prod(static_spatial)) == flat_size:
        concrete_shape = [1] + static_spatial
    else:
        concrete_shape = [1, flat_size]

    # Run inference on every sample
    all_outputs = []
    for x in X_samples:
        x_in = x.flatten().astype(np.float32).reshape(concrete_shape)
        out  = sess.run(None, {input_name: x_in})[0]   
        all_outputs.append(out.flatten())

    all_outputs = np.array(all_outputs, dtype=np.float32)  
    Y_arr       = np.array(Y_samples,   dtype=np.int32)

    header = 'Output distribution analysis (ODL train samples)'
    logger.info(header)

    target_mags_correct = []
    target_mags_wrong   = []

    for c in range(n_classes):
        mask       = (Y_arr == c)
        wrong_mask = (Y_arr != c)
        cname      = class_names[c] if c < len(class_names) else str(c)

        if mask.sum() == 0:
            logger.warning(f'class {c} ({cname}): no samples found, using defaults.')
            target_mags_correct.append(TARGET_MAG_DEFAULT)
            target_mags_wrong.append(-TARGET_MAG_DEFAULT)
            continue

        # Correct-class target: mean of logit[c] on samples where label==c
        correct_logit = all_outputs[mask, c]
        class_correct = int(round(float(correct_logit.mean())))
        target_mags_correct.append(class_correct)

        # Wrong-class target: mean of logit[c] on samples where label!=c
        if wrong_mask.sum() > 0:
            wrong_logit_c = all_outputs[wrong_mask, c]
            class_wrong   = int(round(float(wrong_logit_c.mean())))
        else:
            wrong_logit_c = np.array([], dtype=np.float32)
            class_wrong   = -TARGET_MAG_DEFAULT
        target_mags_wrong.append(class_wrong)

        line_correct = (
            f'correct-class logit:  '
            f'mean={correct_logit.mean()}  std={correct_logit.std()}  '
            f'min={int(correct_logit.min())}  max={int(correct_logit.max())}  '
            f'TARGET_CORRECT={class_correct}'
        )
        if len(wrong_logit_c) > 0:
            line_wrong = (
                f'wrong-class  logit[c]: '
                f'mean={wrong_logit_c.mean()}  std={wrong_logit_c.std()}  '
                f'min={int(wrong_logit_c.min())}  max={int(wrong_logit_c.max())}  '
                f'TARGET_WRONG={class_wrong}'
            )
        else:
            line_wrong = f'wrong-class  logit[c]: N/A  TARGET_WRONG={class_wrong}'

        logger.info(f'class {c} ({cname}):')
        logger.info(f'{line_correct.strip()}')
        logger.info(f'{line_wrong.strip()}')

    summary = (f'TARGET_CORRECT: {target_mags_correct}  TARGET_WRONG: {target_mags_wrong}')
    logger.info(f'{summary.strip()}')
    return target_mags_correct, target_mags_wrong


def export_for_ondevice_training(onnx_path, args, target_mags_correct=None, target_mags_wrong=None):
    """Parse a post QAT ONNX model and write integer_model_config.h and integer_model_config.c.

    Determines the split point based on args.trainable_layers_from_last:
    - If equal to the total number of compute layers, emits a full model
      (PREQUANT + all compute layers).
    - If less, emits only the last N compute layers without PREQUANT.
      The frozen prefix is handled separately by TVM via export_frozen_model().

    Args:
        onnx_path           : path to QAT model.onnx.
        args                : training args. Requires .output_dir and
                              .trainable_layers_from_last.
        target_mags_correct : list[int] of length n_classes, or None to use defaults.
        target_mags_wrong   : list[int] of length n_classes (negative), or None to use defaults.
    """
    out_dir = os.path.join(args.output_dir, 'integer_model_config')
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f'parsing ONNX: {onnx_path}')
    graph    = load_graph(onnx_path)
    segments = segment_graph(graph)
    blocks   = classify_segments(segments)
    layers   = extract_all_blocks(blocks)

    all_compute   = [l for l in layers if l['block_type'] != 'PREQUANT']
    n_all_compute = len(all_compute)
    n_trainable_req = getattr(args, 'trainable_layers_from_last', n_all_compute)

    if n_trainable_req == 0:
        raise ValueError(
            f"trainable_layers_from_last=0. Must be >= 1 to emit any C artifacts."
        )

    if n_trainable_req > n_all_compute:
        logger.warning( f'trainable_layers_from_last={n_trainable_req} exceeds available compute layers ({n_all_compute}). Clamping to {n_all_compute}.')
    n_trainable = min(n_trainable_req, n_all_compute)

    if n_trainable < n_all_compute:
        layers_to_emit = all_compute[-n_trainable:]
        is_full_model  = False
        logger.info(f'split model: emitting last {n_trainable}/{n_all_compute} compute layers. Input is frozen-model uint8 output, no PREQUANT emitted.')
    else:
        layers_to_emit = all_compute
        is_full_model  = True
        logger.info(f'full model: {n_all_compute} compute layers. PREQUANT compiled into TVM frozen model.')

    inp_spatial = [int(d) for d in graph.inputs[0].shape if isinstance(d, int) and d > 0]
    raw_in_size = int(np.prod(inp_spatial)) 

    meta = {
        'input_size':  raw_in_size,
        'n_classes':   layers_to_emit[-1]['out_size'],
        'n_trainable': n_trainable,
    }

    logger.info(f'n_classes={meta["n_classes"]}, trainable={n_trainable}/{n_all_compute}')

    emit_integer_model_config_h(layers_to_emit, meta, out_dir,
                                target_mags_correct=target_mags_correct,
                                target_mags_wrong=target_mags_wrong,
                                is_full_model=is_full_model)
    emit_integer_model_config_c(layers_to_emit, meta, out_dir, is_full_model=is_full_model)


def _emit_identity_onnx(inp_tensor, path):
    """Write a single-op Identity ONNX model to path.

    Used as a passthrough frozen model when the full pipeline runs via integer ODL
    and TVM compilation still expects a frozen model file to exist.

    Args:
        inp_tensor : onnx-graphsurgeon Variable representing the model's input tensor.
        path       : output file path for the ONNX model.
    """
    x   = gs.Variable('input',  dtype=inp_tensor.dtype, shape=inp_tensor.shape)
    out = gs.Variable('output', dtype=inp_tensor.dtype, shape=inp_tensor.shape)
    g   = gs.Graph(nodes=[gs.Node('Identity', inputs=[x], outputs=[out])], inputs=[x], outputs=[out])
    onnx.save(gs.export_onnx(g), path)
    logger.info(f'passthrough Identity frozen model saved: {path}')


def _emit_prequant_onnx(graph, blocks, path):
    """Extract the PREQUANT block as a standalone frozen model for TVM compilation.

    Used for full model ODL where all compute layers are trainable. TVM compiles
    the PREQUANT ops (float input to int8 output) so INTODT receives int8 directly.

    Args:
        graph  : onnx-graphsurgeon Graph (topologically sorted).
        blocks : output of classify_segments() -- list of (name, _, nodes).
        path   : output file path for the ONNX model.
    """
    prequant_block = next((b for b in blocks if b[0] == 'PREQUANT'), None)
    if prequant_block is None:
        raise ValueError(
            'export_frozen_model: no PREQUANT block found in graph. '
            'Cannot build PREQUANT-only frozen model for full model ODL.'
        )
    split_tensor = prequant_block[2][-1].outputs[0]
    frozen_graph = extract_frozen_subgraph(graph, split_tensor)
    onnx.save(gs.export_onnx(frozen_graph), path)
    logger.info(f'PREQUANT-only frozen model saved: {path} '
                f'(output tensor: {split_tensor.name}, shape={split_tensor.shape})')


def export_frozen_model(onnx_path, args):
    """Split a NPU ONNX model and save the frozen prefix subgraph for TVM compilation.

    The split point is the output tensor of the last Clip/Relu node in the last
    frozen compute block. RESHAPE/Flatten ops between the frozen end and the first
    trainable block are not included; the TVM output shape must match what
    INTODT_Forward/Backward expects.

    When trainable_layers_from_last >= total compute layers (full model ODL),
    the PREQUANT block is extracted as a standalone frozen model for TVM. INTODT
    then receives int8 input directly from TVM with no PREQUANT layer of its own.

    Saves to: <args.output_dir>/frozen_model/model.onnx

    Args:
        onnx_path : path to QAT model.onnx.
        args      : training args. Requires .output_dir and
                    .trainable_layers_from_last.
    """
    k = getattr(args, 'trainable_layers_from_last', 1)

    frozen_dir  = os.path.join(args.output_dir, 'frozen_model')
    os.makedirs(frozen_dir, exist_ok=True)
    frozen_path = os.path.join(frozen_dir, 'model.onnx')

    graph    = load_graph(onnx_path)
    segments = segment_graph(graph)
    blocks   = classify_segments(segments)

    # Compute block indices: non-PREQUANT, non-RESHAPE
    compute_indices = [i for i, (name, _, _) in enumerate(blocks)
                       if name not in ('PREQUANT', 'RESHAPE')]
    n_all_compute = len(compute_indices)

    if k >= n_all_compute:
        logger.info(
            f'export_frozen_model: k={k} >= n_all_compute={n_all_compute}. '
            f'Full model ODL. Emitting PREQUANT-only frozen model for TVM.'
        )
        _emit_prequant_onnx(graph, blocks, frozen_path)
        return

    # k < n_all_compute: split at last node output of last frozen compute block.
    # Frozen compute blocks = first (n_all_compute - k).
    last_frozen_compute_pos = n_all_compute - k - 1
    last_frozen_block_idx   = compute_indices[last_frozen_compute_pos]
    last_frozen_seg         = blocks[last_frozen_block_idx][2]

    split_tensor = last_frozen_seg[-1].outputs[0]

    logger.info(
        f'split: frozen {n_all_compute - k}/{n_all_compute} compute layers, trainable={k}. '
        f'Split tensor: {split_tensor.name}  shape={split_tensor.shape}'
    )

    frozen_graph = extract_frozen_subgraph(graph, split_tensor)
    frozen_onnx  = gs.export_onnx(frozen_graph)
    onnx.save(frozen_onnx, frozen_path)
    logger.info(f'frozen model saved: {frozen_path}')


def export_training_data(dataset_train, dataset_val, dataset_test, args, onnx_path=None):
    """Sample data and write train_data.h / train_data.c.

    Samples args.export_samples_per_class examples per class from each dataset.
    Training samples are shuffled before writing. If onnx_path is provided,
    runs analyze_output_distribution() on the exported training samples to derive
    per-class TARGET_MAG values from the model's actual output distribution.

    Args:
        dataset_train : GenericTSDataset for training samples.
        dataset_val   : GenericTSDataset for validation samples.
        dataset_test  : GenericTSDataset for test samples.
        args          : training args. Requires .output_dir and
                        .export_samples_per_class (list [n_train, n_val, n_test]).
        onnx_path     : optional path to QAT model.onnx. If provided and the file
                        exists, distribution analysis is run to derive TARGET_MAG.

    Returns:
        (target_mags_correct, target_mags_wrong) : tuple of two list[int], each of
        length n_classes. Returns defaults if onnx_path is None or not found.
    """
    out_dir = os.path.join(args.output_dir, 'integer_model_config')
    os.makedirs(out_dir, exist_ok=True)

    spec = getattr(args, 'export_samples_per_class', [10, 10, 10])
    if isinstance(spec, str):
        spec = ast.literal_eval(spec)
    n_train, n_val, n_test = int(spec[0]), int(spec[1]), int(spec[2])

    logger.info(f'Export samples per class: train={n_train}, val={n_val}, test={n_test} per class')

    X_train, Y_train = _collect_samples(dataset_train, n_train)
    X_val,   Y_val   = _collect_samples(dataset_val,   n_val)
    X_test,  Y_test  = _collect_samples(dataset_test,  n_test)

    idx = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[idx], Y_train[idx]

    n_classes = len(dataset_train.classes)
    if onnx_path is not None and os.path.isfile(onnx_path):
        target_mags_correct, target_mags_wrong = analyze_output_distribution(
            onnx_path, X_train, Y_train,
            class_names=list(dataset_train.classes)
        )

    emit_train_data_h(len(X_train), len(X_val), len(X_test), out_dir)
    emit_train_data_c( (X_train, Y_train), (X_val,   Y_val), (X_test,  Y_test), out_dir)

    logger.info(f'Training data written to {out_dir} ({len(X_train)+len(X_val)+len(X_test)} total samples)')
    return target_mags_correct, target_mags_wrong
