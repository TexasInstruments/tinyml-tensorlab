import os
import onnx
import numpy as np
from logging import getLogger
from glob import glob
import re
import shutil
import subprocess
import tempfile

def _rearrange_fb_weights(fb_weights_list, op_ch, ip_ch, sublayer_size):
    packed = []
    for i in range(0, op_ch, sublayer_size):
        sublayer = fb_weights_list[i:i+sublayer_size]
        rows = []
        for ip_row in range(0, ip_ch, 4):
            for op_row in range(0, sublayer_size, 4):
                vec = (sublayer[op_row+3][ip_row:ip_row+4] +
                       sublayer[op_row+2][ip_row:ip_row+4] +
                       sublayer[op_row+1][ip_row:ip_row+4] +
                       sublayer[op_row  ][ip_row:ip_row+4])
                rows.append(vec)
        packed.append(rows)
    return packed


def _fmt_2bit(row):
    s = "0b"
    for v in row:
        v = int(v)
        if   v ==  0: s += "00"
        elif v ==  1: s += "01"
        elif v == -1: s += "11"
        else: raise ValueError(f"Invalid 2-bit weight: {v}")
    return [s + ","]


def _fmt_4bit(row):
    s = ""
    for i in range(0, len(row), 4):
        w = [int(x) for x in row[i:i+4]]
        w = [v + 16 if v < 0 else v for v in w]
        b = [f"{v:04b}" for v in w]
        s += b[0][:2]+b[1][:2]+b[2][:2]+b[3][:2]
        s += b[0][2:]+b[1][2:]+b[2][2:]+b[3][2:]
    return [f"0b{s[32:]},", f"0b{s[:32]},"]


def _fmt_8bit(row):
    s = ""
    for i in range(0, len(row), 4):
        w = [int(x) for x in row[i:i+4]]
        w = [v + 256 if v < 0 else v for v in w]
        b = [f"{v:08b}" for v in w]
        s += b[0][:2]+b[1][:2]+b[2][:2]+b[3][:2]
        s += b[0][2:4]+b[1][2:4]+b[2][2:4]+b[3][2:4]
        s += b[0][4:6]+b[1][4:6]+b[2][4:6]+b[3][4:6]
        s += b[0][6:]+b[1][6:]+b[2][6:]+b[3][6:]
    return [f"0b{s[96:]},", f"0b{s[64:96]},", f"0b{s[32:64]},", f"0b{s[:32]},"]


_BW_CFG = {
    2: {'sublayer_size': 16, 'ctl0_msb': 0x12019, 'ctl0_lsb': 0x12011, 'fmt': _fmt_2bit, 'words_per_row': 1},
    4: {'sublayer_size':  8, 'ctl0_msb': 0x13019, 'ctl0_lsb': 0x13011, 'fmt': _fmt_4bit, 'words_per_row': 2},
    8: {'sublayer_size':  4, 'ctl0_msb': 0x14019, 'ctl0_lsb': 0x14011, 'fmt': _fmt_8bit, 'words_per_row': 4},
}

_DUMMY_TMA_INIT_WORDS = 2

_FB_INS_WORDS = [
    '0x18160800', '0x0026', '0x01031800', '0x0000',
    '0x00002800', '0x0000', '0x00003800', '0x0000',
    '0xb8000000', '0x0008', '0x10820420', '0x0002',
    '0x00000000', '0x0000', '0x10001400', '0x0042',
    '0x00000000', '0x4240', '0x10820520', '0x2442',
    '0x00000000', '0x6640', '0x10001400', '0x1042',
    '0x00000000', '0x5240', '0x00000000', '0x3440',
    '0x00000001', '0x7640', '0x0000008c', '0x0000',
]


def _write_const_array(c_path, sym, typ, values, val_fmt=str):
    with open(c_path, 'w') as f:
        f.write('#include "fe_model.h"\n\n')
        f.write(f'const {typ} {sym}[] = {{\n')
        for v in values:
            f.write(f'    {val_fmt(v)},\n')
        f.write('};\n\n')
        f.write(f'const uint32_t {sym}_LEN = (uint32_t)(sizeof({sym}) / sizeof({sym}[0]));\n')


def pack_and_compile_filterbank(fb_onnx_path, output_dir, bitwidth=2, stride_t=4, branched_bits=16,
                                cross_compiler=None, cross_compiler_options=None):
    """
    Pack filterbank ONNX weights and compile into fe_model.a + fe_model.h + model_autogen.h.

    Unlike the earlier packer, parameter images are not exposed as individual global
    arrays. Instead each 16-output-channel block's packed image is emitted as
    a static array and referenced through a single exported descriptor table
    (g_fb_param_slices), matching the FB_ParamSlice ABI consumed by the SDK.
    Kernel-dimension splitting is intentionally not implemented (one full
    param image per output block) per existing hardware/firmware convention.
    """
    import subprocess
    from onnx import numpy_helper as nh

    if bitwidth not in _BW_CFG:
        raise ValueError(f"filterbank_weight_bits must be 2, 4, or 8; got {bitwidth}")

    logger = getLogger("root.run_filterbank_compilation")
    bw            = _BW_CFG[bitwidth]
    sublayer_size = bw['sublayer_size']
    fmt           = bw['fmt']
    ctl0_msb      = bw['ctl0_msb']
    ctl0_lsb      = bw['ctl0_lsb']

    model       = onnx.load(fb_onnx_path)
    keys        = [init.name for init in model.graph.initializer]
    raw         = {init.name: nh.to_array(init) for init in model.graph.initializer}
    fb_weights  = raw[keys[0]].squeeze()
    fb_offset   = raw[keys[1]].squeeze()
    fb_scale    = raw[keys[2]].squeeze()
    fb_shift    = np.log2(1.0 / raw[keys[3]].squeeze())

    op_ch, ip_ch = fb_weights.shape[0], fb_weights.shape[1]
    if op_ch % sublayer_size != 0:
        raise ValueError(f"op_ch={op_ch} must be divisible by sublayer_size={sublayer_size}")
    if ip_ch % 8 != 0:
        raise ValueError(f"ip_ch={ip_ch} must be divisible by 8")

    num_output_blocks = op_ch // sublayer_size
    kernel_slice_t    = ip_ch      # no kernel-dimension splitting
    num_kernel_slices = 1

    packed = _rearrange_fb_weights(fb_weights.tolist(), op_ch, ip_ch, sublayer_size)

    block_word_lines = []
    for rows in packed:
        lines = []
        for row in rows:
            lines.extend(fmt(row))
        block_word_lines.append(lines)

    rows_per_slice          = len(block_word_lines[0])
    max_params_words        = rows_per_slice
    params_words_per_slice  = rows_per_slice + _DUMMY_TMA_INIT_WORDS
    params_load_capacity    = params_words_per_slice
    arbias0_row_word64      = rows_per_slice // 2
    lc2_per_slice           = (ip_ch // 8) - 1

    os.makedirs(output_dir, exist_ok=True)

    # --- model_autogen.h (written first: fe_model.c/.h depend on its macros) ---
    autogen_path = os.path.join(output_dir, 'model_autogen.h')
    with open(autogen_path, 'w') as f:
        f.write('#ifndef MODEL_AUTOGEN_H\n#define MODEL_AUTOGEN_H\n\n')
        f.write('/* ============================================================\n')
        f.write(' * Full model geometry\n')
        f.write(' * ============================================================ */\n')
        f.write(f'#define MODEL_IN_CHANNELS            (1u)\n')
        f.write(f'#define MODEL_CONV_OUT_CHANNELS      ({op_ch}u)\n')
        f.write(f'#define MODEL_CONV_KERNEL_T          ({ip_ch}u)\n')
        f.write(f'#define MODEL_CONV_STRIDE_T          ({stride_t}u)\n')
        f.write(f'#define MODEL_MAX_OUT_CH_PER_ITER    ({sublayer_size}u)\n')
        f.write(f'#define MODEL_FBANK_LEN              (MODEL_CONV_OUT_CHANNELS)\n')
        f.write(f'#define MODEL_WEIGHT_BITS            ({bitwidth}u)\n')
        f.write(f'#define MODEL_BRANCHED_BITS          ({branched_bits}u)\n\n')
        f.write('/* ============================================================\n')
        f.write(' * Execution slice geometry\n')
        f.write(' * ============================================================ */\n')
        f.write(f'#define MODEL_NUM_OUTPUT_BLOCKS      ({num_output_blocks}u)\n')
        f.write(f'#define MODEL_NUM_KERNEL_SLICES      ({num_kernel_slices}u)\n')
        f.write(f'#define MODEL_PARAMS_LOAD_CAPACITY   ({params_load_capacity}u)\n')
        f.write(f'#define MODEL_ARBIAS0_ROW_WORD64     ({arbias0_row_word64}u)\n\n')
        f.write('/* CTL0 values */\n\n')
        f.write(f'#define MODEL_FE_CTL0_MSB (0x{ctl0_msb:05x}u)\n')
        f.write(f'#define MODEL_FE_CTL0_LSB (0x{ctl0_lsb:05x}u)\n\n')
        f.write('#endif /* MODEL_AUTOGEN_H */\n')

    # --- fe_model.h (single umbrella header) ---
    h_path = os.path.join(output_dir, 'fe_model.h')
    with open(h_path, 'w') as f:
        f.write('#ifndef FE_MODEL_H\n#define FE_MODEL_H\n\n')
        f.write('#include <stdint.h>\n#include "model_autogen.h"\n\n')
        f.write('typedef struct\n{\n')
        f.write('    const uint32_t *params;\n')
        f.write('    uint32_t words;\n')
        f.write('    uint16_t mmr0_offset_words;\n')
        f.write('} FB_ParamSlice;\n\n')
        f.write('extern const FB_ParamSlice g_fb_param_slices[MODEL_NUM_OUTPUT_BLOCKS][MODEL_NUM_KERNEL_SLICES];\n\n')
        f.write('extern const uint32_t FB_INS[];\n')
        f.write('extern const uint32_t FB_INS_LEN;\n\n')
        f.write('extern const uint32_t FBANK_MMR[];\n')
        f.write('extern const uint32_t FBANK_MMR_LEN;\n\n')
        f.write('extern const int16_t FB_OFFSET[];\n')
        f.write('extern const uint32_t FB_OFFSET_LEN;\n\n')
        f.write('extern const uint8_t FB_SCALE[];\n')
        f.write('extern const uint32_t FB_SCALE_LEN;\n\n')
        f.write('extern const uint8_t FB_SHIFT[];\n')
        f.write('extern const uint32_t FB_SHIFT_LEN;\n\n')
        f.write('#endif /* FE_MODEL_H */\n')

    c_files = []

    # --- fb_params_all.c: static per-block images + exported descriptor table ---
    params_path = os.path.join(output_dir, 'fb_params_all.c')
    c_files.append(params_path)
    with open(params_path, 'w') as f:
        f.write('#include "fe_model.h"\n\n')
        block_syms = []
        for i, lines in enumerate(block_word_lines):
            sym = f'FB_PARAMS_BLOCK{i}_SLICE0'
            block_syms.append(sym)
            f.write(f'static const uint32_t {sym}[] = {{\n')
            for line in lines:
                f.write(f'    {line}\n')
            f.write('    0b00000000000000000000000000000000,\n' * _DUMMY_TMA_INIT_WORDS)
            f.write('};\n\n')

        f.write('const FB_ParamSlice g_fb_param_slices[MODEL_NUM_OUTPUT_BLOCKS][MODEL_NUM_KERNEL_SLICES] = {\n')
        for sym in block_syms:
            f.write('    {\n')
            f.write(f'        {{ {sym}, {params_words_per_slice}u, 0u }},\n')
            f.write('    },\n')
        f.write('};\n')

    # --- remaining const arrays, each with a companion _LEN ---
    fb_ins_path = os.path.join(output_dir, 'fb_ins.c')
    c_files.append(fb_ins_path)
    _write_const_array(fb_ins_path, 'FB_INS', 'uint32_t', _FB_INS_WORDS, lambda v: v)

    mmr = (['0x0'] * 3 + ['0xff0000', '0x8000800', f'0x{ctl0_msb:05x}'] + ['0x0'] * 2 +
           [f'0x{lc2_per_slice:08x}'] + ['0x0'] * 16 + [f'0x{arbias0_row_word64:08x}'] + ['0x0'] * 8)
    fb_mmr_path = os.path.join(output_dir, 'fb_mmr.c')
    c_files.append(fb_mmr_path)
    _write_const_array(fb_mmr_path, 'FBANK_MMR', 'uint32_t', mmr, lambda v: v)

    fb_offset_path = os.path.join(output_dir, 'fb_offset.c')
    c_files.append(fb_offset_path)
    _write_const_array(fb_offset_path, 'FB_OFFSET', 'int16_t', fb_offset, lambda v: str(int(v)))

    fb_scale_path = os.path.join(output_dir, 'fb_scale.c')
    c_files.append(fb_scale_path)
    _write_const_array(fb_scale_path, 'FB_SCALE', 'uint8_t', fb_scale, lambda v: str(int(v)))

    fb_shift_path = os.path.join(output_dir, 'fb_shift.c')
    c_files.append(fb_shift_path)
    _write_const_array(fb_shift_path, 'FB_SHIFT', 'uint8_t', fb_shift, lambda v: str(int(v)))

    with open(os.path.join(output_dir, 'FB_layer_metadata.txt'), 'w') as f:
        f.write(f"Output blocks: {num_output_blocks} x {sublayer_size} op_ch, ip_ch={ip_ch}, op_ch={op_ch}\n")
        f.write(f"Kernel slices per block: {num_kernel_slices} (no kernel-dimension splitting)\n")
        f.write(f"Params words per slice: {params_words_per_slice}\n")
        f.write(f"ARBIAS0 row (word64): {arbias0_row_word64}\n")
        f.write(f"LC2 loop iters: {lc2_per_slice + 1}\n")
        f.write(f"CTL0 MSB: 0x{ctl0_msb:05x}  CTL0 LSB: 0x{ctl0_lsb:05x}\n")

    if not (cross_compiler and (os.path.exists(cross_compiler) or os.path.exists(cross_compiler + '.exe'))):
        print(f"Cross-compiler not found — sources written to {output_dir}, skipping compile.")
        return h_path, None

    flags     = (cross_compiler_options or '').split()
    obj_files = []
    for c_path in c_files:
        obj = c_path[:-2] + '.o'
        r   = subprocess.run(
            [cross_compiler, '-c', c_path, '-I', output_dir, '-o', obj] + flags,
            capture_output=True, text=True
        )
        if r.returncode != 0:
            raise RuntimeError(f"Compile failed for {os.path.basename(c_path)}:\n{r.stderr}")
        obj_files.append(obj)

    bin_dir = os.path.dirname(cross_compiler)
    ar_names = ('tiarmar', 'llvm-ar', 'tiarmar.exe', 'llvm-ar.exe')
    ar = next(
        (os.path.join(bin_dir, n) for n in ar_names
         if os.path.exists(os.path.join(bin_dir, n))),
        None
    )
    if ar is None:
        raise RuntimeError(f"No archiver (tiarmar/llvm-ar) found in {bin_dir}")

    lib_path = os.path.join(output_dir, 'fe_model.a')
    r = subprocess.run([ar, 'rcs', lib_path] + obj_files, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ar failed:\n{r.stderr}")

    extensions = ('*.c', '*.o', '*.txt')
    for ext in extensions:
        files = glob(os.path.join(output_dir, ext))
        for filename in files:
            os.remove(filename)
    logger.info(f"filterbank artifacts located at: {output_dir}")
    return h_path, lib_path

def run_filterbank_compilation(tvm_onnx_path, compilation_root_dir, bw, stride, branched_bits, cross_compiler, cross_compiler_options):
    logger = getLogger("root.run_filterbank_compilation")
    fb_folder = os.path.dirname(tvm_onnx_path)
    fb_onnx_path = os.path.join(fb_folder, 'model_fb.onnx')
    fb_pack_dir = os.path.join(compilation_root_dir, 'fe_model')
    try:
        pack_and_compile_filterbank(
            fb_onnx_path=fb_onnx_path,
            output_dir=fb_pack_dir,
            bitwidth=bw,
            stride_t=stride,
            branched_bits=branched_bits,
            cross_compiler=cross_compiler,
            cross_compiler_options=cross_compiler_options,
        )
    except Exception as e:
        import traceback
        logger.warning(f"Filterbank packing failed: {e}")
        traceback.print_exc()

_CALL_PATTERN = re.compile(
    r'^(?P<indent>[ \t]*)tvmgen_default_fused_layout_transform\(\s*'
    r'(?P<src>\w+)\s*,\s*(?P<mid1>\w+)\s*,\s*(?P<ws1>\w+)\s*\)\s*;\s*\n'
    r'[ \t]*tvmgen_default_ti_npu_zeroPaddingNHWC\w*\(\s*'
    r'(?P<mid2>\w+)\s*,\s*(?P<dst>\w+)\s*,\s*(?P<ws2>\w+)\s*\)\s*;',
    re.MULTILINE,
)
_INPUT_SHAPE_PATTERN = re.compile(r'Inputs:\s*\n\s*\*\s*Tensor\[\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)')


def _get_fb_output_geometry(artifacts_dir):
    header_path = os.path.join(artifacts_dir, 'tvmgen_default.h')
    with open(header_path, 'r') as f:
        header_text = f.read()
    match = _INPUT_SHAPE_PATTERN.search(header_text)
    if not match:
        raise RuntimeError(f"Could not find model input tensor shape in {header_path}")
    _, channels, features, _ = (int(g) for g in match.groups())
    return channels, features


def _patch_lib1(artifacts_dir, channels, features):
    lib1_path = os.path.join(artifacts_dir, 'lib1.c')
    with open(lib1_path, 'r') as f:
        text = f.read()

    match = _CALL_PATTERN.search(text)
    if not match:
        raise RuntimeError(
            f"Expected layout_transform + zeroPaddingNHWC call pair not found in {lib1_path}; "
            "refusing to produce mod.a without applying the required hardware workaround.")
    if match.group('mid1') != match.group('mid2'):
        raise RuntimeError(
            f"layout_transform/zeroPaddingNHWC intermediate buffers don't match "
            f"({match.group('mid1')} vs {match.group('mid2')}) in {lib1_path}; pattern assumption violated.")

    size = channels * (features + 2)
    indent = match.group('indent')
    replacement = (
        f"{indent}#if 0\n"
        f"{match.group(0)}\n"
        f"{indent}#else\n"
        f"{indent}memcpy({match.group('dst')}, {match.group('src')}, {size});"
        f"  // {channels} = fb_output_channel | {features + 2} = (fb_output_features + 2)\n"
        f"{indent}#endif"
    )
    text = text[:match.start()] + replacement + text[match.end():]
    with open(lib1_path, 'w') as f:
        f.write(text)


def _find_archiver(cross_compiler):
    bin_dir = os.path.dirname(cross_compiler)
    for name in ('tiarmar', 'llvm-ar', 'tiarmar.exe', 'llvm-ar.exe'):
        candidate = os.path.join(bin_dir, name)
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(f"No archiver (tiarmar/llvm-ar) found in {bin_dir}")


def _tvm_include_dirs():
    import tvm
    tvm_root = os.path.dirname(os.path.abspath(tvm.__file__))
    return [
        os.path.join(tvm_root, 'standalone_crt', 'include'),
        os.path.join(tvm_root, '3rdparty', 'tinie-api'),
    ]


_LIB_SOURCE_PATTERN = re.compile(r'^lib\d+\.c$')


def _rebuild_mod_a(artifacts_dir, mod_a_path, cross_compiler, cross_compiler_options):
    archiver = _find_archiver(cross_compiler)
    lib_sources = sorted(
        p for p in glob(os.path.join(artifacts_dir, 'lib*.c'))
        if _LIB_SOURCE_PATTERN.match(os.path.basename(p))
    )
    flags = (cross_compiler_options or '').split()
    include_dirs = [artifacts_dir] + _tvm_include_dirs()

    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([archiver, 't', mod_a_path], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to list members of {mod_a_path}:\n{r.stderr}")
        members = [m for m in r.stdout.splitlines() if m.strip()]
        lib_object_names = {os.path.splitext(os.path.basename(s))[0] + '.obj' for s in lib_sources}
        preserved = [m for m in members if os.path.basename(m) not in lib_object_names]

        preserved_paths = []
        if preserved:
            r = subprocess.run([archiver, 'x', mod_a_path] + preserved, cwd=tmp, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"Failed to extract {preserved} from {mod_a_path}:\n{r.stderr}")
            preserved_paths = [os.path.join(tmp, m) for m in preserved]

        obj_files = []
        for src in lib_sources:
            obj_path = os.path.join(tmp, os.path.splitext(os.path.basename(src))[0] + '.o')
            include_args = []
            for inc in include_dirs:
                include_args += ['-I', inc]
            r = subprocess.run(
                [cross_compiler, '-c', src, '-o', obj_path] + include_args + flags,
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f"Compile failed for {os.path.basename(src)}:\n{r.stderr}")
            obj_files.append(obj_path)

        new_mod_a = os.path.join(tmp, os.path.basename(mod_a_path))
        r = subprocess.run([archiver, 'rcs', new_mod_a] + preserved_paths + obj_files, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ar failed:\n{r.stderr}")

        shutil.copyfile(new_mod_a, mod_a_path)


def apply_fb_zero_pad(artifacts_dir, mod_a_path, cross_compiler, cross_compiler_options):
    logger = getLogger("root.apply_fb_zero_pad")
    if not os.path.exists(mod_a_path):
        logger.warning(f"apply_fb_zero_pad: {mod_a_path} not found (output_format != 'a'?); skipping.")
        return
    channels, features = _get_fb_output_geometry(artifacts_dir)
    _patch_lib1(artifacts_dir, channels, features)
    _rebuild_mod_a(artifacts_dir, mod_a_path, cross_compiler, cross_compiler_options)
