#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
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
"""
Memory pre-flight check.

Exports a freshly-created (untrained) model to ONNX and runs it through the
exact same compile path the end-of-pipeline compilation stage uses, purely to
read off the ROM/RAM footprint before spending time on training. ROM/RAM is
determined by architecture shape/dtype, not by trained weight values, so this
works fine on a randomly-initialized model.

Both `compilation.py` (NN model) and `fel_memory/compilation_fel.py` (Feature
Extraction Library) are used as-is, unmodified -- this module only wraps them
and scrapes their combined stdout/stderr for the two clean memory-report
blocks they already print, discarding the much larger torrent of TVM/compiler
diagnostics around them.
"""

import contextlib
import logging
import os
import re
import sys
from logging import getLogger

import torch

from tinyml_tinyverse.common.utils import utils
from tinyml_tinyverse.references.common import compilation
from tinyml_tinyverse.references.common.fel_memory import compilation_fel

_logger = getLogger("root.memory_preflight")

_MODEL_HEADER = "TI Model Library Memory Usage"
_FEL_HEADER = "TI Feature Extraction Library Memory Usage"

_FULL_LOG_NAME = "memory_preflight_full.log"

_BANNER_TITLE = "===== MEMORY PRE-FLIGHT ESTIMATE ====="


def _block_regex(header_text):
    """Build a regex matching one memory-report block.

    Blocks look like (see compilation_fel.py:print_memory_report / the
    ti_mcu_nnc logger emitting the model-side equivalent)::

        ============================================================
        TI Model Library Memory Usage (mod.a)
        ============================================================
        Code:                      1176 bytes (    1.15 KB)
        RO Data:                   1616 bytes (    1.58 KB)
        RW Data:                   5296 bytes (    5.17 KB)
        Total:                     8088 bytes (    7.90 KB)
        ============================================================

    Lines may carry a logger-added prefix (timestamps, logger name, etc.), so
    we match "line containing a run of '=' chars" rather than an exact banner
    string, and stop consuming "content" lines as soon as a "Total:" line is
    seen (non-greedy) followed by the closing banner.
    """
    return re.compile(
        r"^.*={10,}.*\n"
        r"^.*" + re.escape(header_text) + r".*\n"
        r"^.*={10,}.*\n"
        r"(?:^(?!.*={10,}).*\n)*?"
        r"^.*Total:.*\n"
        r"^.*={10,}.*$",
        re.MULTILINE,
    )


def _parse_bytes(block_text, label):
    m = re.search(re.escape(label) + r":\s*(\d+)\s*bytes", block_text)
    return int(m.group(1)) if m else None


def _extract_report(full_output, header_text):
    """Return (block_text_or_None, parsed_dict_or_None) for one report block."""
    match = _block_regex(header_text).search(full_output)
    if not match:
        return None, None
    block_text = match.group(0)
    try:
        parsed = {
            "code": _parse_bytes(block_text, "Code"),
            "ro_data": _parse_bytes(block_text, "RO Data"),
            "rw_data": _parse_bytes(block_text, "RW Data"),
            "total": _parse_bytes(block_text, "Total"),
        }
        if any(v is None for v in parsed.values()):
            parsed = None
    except Exception:
        parsed = None
    return block_text, parsed


@contextlib.contextmanager
def redirect_os_level_output(target_path):
    """Redirect the OS-level fd 1/fd 2 (not just sys.stdout/sys.stderr) into
    `target_path` for the duration of the block, restoring the originals in
    a `finally`.

    os.system() subprocess calls (used by compilation_fel.py's per-device
    functions) and TVM's native C++ runtime (used by compilation.py's
    drive_compile) write directly to the process's inherited file
    descriptors -- contextlib.redirect_stdout only swaps the Python-level
    sys.stdout object and does not affect either of those, so the fds
    themselves have to be duped/restored instead.

    Truncates target_path (does not append): _extract_report()'s regex takes
    the FIRST matching report block in the file, so if a caller retries
    estimate_memory() against the same output_dir/log path (e.g. the
    candidate-shape fallback loop, or re-running with a different
    `quantization` value), a stale block left over from an earlier,
    unrelated call must not still be there to be matched by mistake.

    Skipped entirely when fd 1 is a live console (not already redirected to
    a file/pipe by the caller's shell): on Windows, dup2()-ing a real console
    handle away and back is fragile -- e.g. via a plain (unpiped) `.bat`
    invocation, this has been observed to hard-crash the process with no
    Python traceback, while the exact same run succeeds when stdout is piped
    (e.g. `| Tee-Object`) so fd 1 is already a file/pipe handle before this
    ever runs. estimate_memory() degrades to a logged warning on any failure
    anyway, so skipping the capture here just means the raw compiler output
    prints straight to the console instead of being scraped into
    target_path -- _extract_report() then finds no block and callers see the
    same "could not find the NN model memory report" warning they'd get from
    any other estimate_memory() failure.
    """
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    if os.isatty(stdout_fd) or os.isatty(stderr_fd):
        yield
        return
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    with open(target_path, "w+b") as target_file:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(target_file.fileno(), stdout_fd)
            os.dup2(target_file.fileno(), stderr_fd)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


@contextlib.contextmanager
def _preserve_root_logger():
    """compilation.py's main() calls mdcl_utils.Logger(name="root", ...),
    which unconditionally does `logger.handlers = []` and reassigns a new
    console/file handler set on the process-wide "root"-named logger.

    That's fine when compilation.py runs as its own standalone script/process
    (the real end-of-pipeline compilation stage), but here we're calling it
    in-process, mid-training-startup, after train_base.setup_training_environment()
    has already pointed that same "root" logger at the real training run.log.
    Without saving/restoring around the call, every subsequent training log
    line would silently end up in the preflight's own compilation.lis instead.
    """
    root_logger = logging.getLogger("root")
    saved_handlers = list(root_logger.handlers)
    saved_level = root_logger.level
    try:
        yield
    finally:
        root_logger.handlers = saved_handlers
        root_logger.setLevel(saved_level)


def estimate_memory(model, input_shape, output_dir, target, cross_compiler,
                     cross_compiler_options, target_c_mcpu, opset_version=17,
                     quantization=0, quantization_method='QAT', weight_bitwidth=8,
                     activation_bitwidth=8, fel_config=None, logger=None):
    """Estimate deployed ROM/RAM footprint of `model` (and optionally the FEL)
    before training starts, using the exact compile path the end-of-pipeline
    compilation stage uses.

    quantization: a TinyMLQuantizationVersion value (0=NO_QUANTIZATION,
    1=QUANTIZATION_GENERIC, 2=QUANTIZATION_TINPU), matching train_base.py's
    own --quantization flag -- NOT a bitwidth. When non-zero, `model` is first
    run through utils.quantization_wrapped_model() with auto_quantization=False
    (fixed weight_bitwidth/activation_bitwidth, no bitwidth search) before
    ONNX export, exactly like the real training pipeline does before its own
    post-training utils.export_model() call -- export_model() itself expects
    an already-wrapped module (with a .convert() method) whenever quantization
    is truthy, it does not wrap on its own.

    fel_config: optional (get_memory_fn_name, cgt_path, device_str) tuple,
    matching tinyml_benchmark.py's ModelCompilation.get_device_fel_function()
    entries, e.g. ('get_memory_c28', constants.C2000_CG_ROOT, 'f28p55x').
    get_memory_fn_name is looked up via getattr on compilation_fel and called
    as fn(output_dir, device_str, cgt_path) -- i.e. `output_dir` doubles as
    the "modelmaker run directory" the real pipeline passes as
    project_run_path. Those get_memory_* functions hard-require
    `<output_dir>/compilation/artifacts` (populated by the NN-model compile
    step below) and a pre-existing
    `<output_dir>/training/quantization/golden_vectors/user_input_config.h`
    -- the latter reflects real feature-extraction config that doesn't exist
    yet at model-creation time, so it is the CALLER's responsibility to have
    put one there if FEL numbers are wanted; if it's missing the FEL compile
    will simply fail and this function degrades to a logged warning.

    Never raises -- any failure degrades to a logged warning so a pre-flight
    problem can never block or crash the real training run.

    Returns {"model": {"code", "ro_data", "rw_data", "total"} or None,
             "fel": {...} or None,
             "model_block": raw_block_string or None,
             "fel_block": raw_block_string or None}.
    """
    log = logger or _logger
    result = {"model": None, "fel": None}
    try:
        os.makedirs(output_dir, exist_ok=True)
        full_log_path = os.path.join(output_dir, _FULL_LOG_NAME)

        onnx_export_dir = os.path.join(output_dir, "onnx_export")
        os.makedirs(onnx_export_dir, exist_ok=True)
        compile_dir = os.path.join(output_dir, "compilation")
        os.makedirs(compile_dir, exist_ok=True)

        with _preserve_root_logger(), redirect_os_level_output(full_log_path):
            # Untrained/random weights are fine here -- ROM/RAM is determined
            # by architecture shape/dtype, not by weight values. Pass through the
            # quantization setting so we can estimate both base (quantization=0)
            # and quantized (quantization>0) memory footprints. generic_model=True
            # keeps the file plaintext.
            try:
                export_model_input = model
                if quantization:
                    # export_model() requires an already quantization_wrapped_model()
                    # (needs a .convert() method) whenever quantization is truthy --
                    # it does not wrap raw modules itself. No real dataset exists at
                    # this point (this may run standalone with no training data), so
                    # auto_quantization=False with the fixed weight/activation
                    # bitwidths given -- no calibration_dataloader/binary search.
                    example_inputs = torch.rand(size=input_shape)
                    export_model_input = utils.quantization_wrapped_model(
                        model, quantization=quantization, quantization_method=quantization_method,
                        weight_bitwidth=weight_bitwidth, activation_bitwidth=activation_bitwidth,
                        epochs=1, output_int=True, auto_quantization=False, example_inputs=example_inputs)
                    # A real training run populates the QAT/PTQ observers' scale
                    # and zero-point across many forward passes over real data
                    # before convert()/export() is ever called. There is no
                    # dataset here, so run a few forward passes on the same
                    # random example_inputs -- ROM/RAM is determined by
                    # architecture shape/dtype, not observed statistics, so
                    # the exact calibration values don't matter for this
                    # purpose, only that observers are populated at all
                    # (an uncalibrated observer leaves scale/zero-point at
                    # their uninitialized default, which convert() rejects).
                    export_model_input.eval()
                    with torch.no_grad():
                        for _ in range(2):
                            export_model_input(example_inputs)
                utils.export_model(export_model_input, input_shape=input_shape, output_dir=onnx_export_dir,
                                    opset_version=opset_version, quantization=quantization, generic_model=True)
                onnx_file = os.path.join(onnx_export_dir, "model.onnx")
            except Exception as exc:
                onnx_file = None
                print(f"[memory_preflight] ONNX export failed: {exc}")

            if onnx_file is not None:
                try:
                    compile_args = compilation.get_args_parser().parse_args([
                        "--FILE", onnx_file,
                        "--output_dir", compile_dir,
                        "--target", target,
                        "--cross_compiler", cross_compiler,
                        "--cross_compiler_options", cross_compiler_options,
                        "--target_c_mcpu", target_c_mcpu,
                        # The ONNX file above is always plaintext (generic_model=True
                        # at export time) regardless of the real run's own
                        # args.generic_model -- compilation.main() would otherwise
                        # try to Fernet-decrypt a plaintext file and fail/corrupt it.
                        "--generic-model", "True",
                    ])
                    compilation.run(compile_args)
                except Exception as exc:
                    print(f"[memory_preflight] NN model compile step raised: {exc}")

            if fel_config is not None:
                fn_name, cgt_path, device_str = fel_config
                try:
                    fel_fn = getattr(compilation_fel, fn_name)
                    fel_fn(output_dir, device_str, cgt_path)
                except Exception as exc:
                    print(f"[memory_preflight] FEL compile step raised: {exc}")

        with open(full_log_path, "r", errors="replace") as fp:
            full_output = fp.read()

        model_block, model_parsed = _extract_report(full_output, _MODEL_HEADER)
        result["model"] = model_parsed
        if model_block is None:
            log.warning(f"Memory pre-flight: could not find the NN model memory report in the "
                        f"compiler output (compile may have failed). See {full_log_path} for details.")

        fel_block = None
        if fel_config is not None:
            fel_block, fel_parsed = _extract_report(full_output, _FEL_HEADER)
            result["fel"] = fel_parsed
            if fel_block is None:
                log.warning(f"Memory pre-flight: could not find the FEL memory report in the "
                            f"compiler output (device unsupported or compile failed). "
                            f"See {full_log_path} for details.")

        # Instead of printing here, we return the raw blocks so the caller can format the output as desired.
        pass

    except Exception as exc:
        log.warning(f"Memory pre-flight check failed and was skipped (training will proceed "
                    f"normally): {exc}")
    return result
