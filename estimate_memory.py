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
Standalone memory pre-flight CLI.

Given a ModelMaker YAML config, resolve the model architecture it describes
(model + feature-extraction preset + target device) and hand a freshly
instantiated (untrained) copy of that model to tinyml_tinyverse's
memory_preflight.estimate_memory() to get the same "Code / RO Data / RW Data /
Total" ROM/RAM report the full pipeline produces at the very end of
training + compilation -- but in seconds, with no dataset download, no
training epochs, and no full TVM/FEL pipeline run other than the two compile
steps needed for the report itself.

Usage:
    python tinyml_modelzoo/estimate_memory.py <config.yaml> [options]
    ./estimate_memory.sh <config.yaml> [options]

This intentionally mirrors the config-resolution steps run_tinyml_modelmaker.py
performs (model description lookup, dataset/feature-extraction/compilation
preset merge) but stops well short of ModelRunner.prepare()/.run() -- no
download, no dataset load, no training, no full compilation.
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import site

# Add necessary directories to sys.path so we can import required modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tinyml-modelmaker')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tinyml-modeloptimization')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tinyml-modeloptimization', 'torchmodelopt')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tinyml-tinyverse')))
# Add user site-packages to ensure tvm and other user-installed packages are found
sys.path.insert(0, site.getusersitepackages())

import yaml

logger = logging.getLogger('root.estimate_memory')


#################################################################################
# Device -> Feature Extraction Library (FEL) compile config.
#
# This is a deliberate, small, manually-kept-in-sync duplicate of
# ModelCompilation.get_device_fel_function() in
# tinyml_modelmaker/ai_modules/common/compilation/tinyml_benchmark.py.
# That method is an instance method on a class whose __init__ resolves a
# full compilation working-directory layout (compilation_path, work_dir,
# package_dir, ...) that this fast/standalone preflight has no reason to
# construct -- duplicating the ~15-line static mapping here is far less
# invasive than instantiating ModelCompilation just to call one lookup.
#################################################################################
def _get_device_fel_function(ai_target_module, device_name):
    constants = ai_target_module.constants
    devices = {
        'AM13E2': ('get_memory_am13', constants.ARM_LLVM_CGT_PATH, 'am13e230x'),
        'MSPM0G5187': ('get_memory_mspm0', constants.ARM_LLVM_CGT_PATH, 'mspm0g5187x'),
        'MSPM0G3507': ('get_memory_mspm0', constants.ARM_LLVM_CGT_PATH, 'mspm0g3507x'),
        'F28E12': ('get_memory_c28', constants.C2000_CG_ROOT, 'f28e12x'),
        'F28P55': ('get_memory_c28', constants.C2000_CG_ROOT, 'f28p55x'),
        'F28P65': ('get_memory_c28', constants.C2000_CG_ROOT, 'f28p65x'),
        'F28P551': ('get_memory_c28', constants.C2000_CG_ROOT, 'f28p551x'),
        'F2807': ('get_memory_c28', constants.C2000_CG_ROOT, 'f2807x'),
        'F2837': ('get_memory_c28', constants.C2000_CG_ROOT, 'f2837xd'),
        'F2838': ('get_memory_c28', constants.C2000_CG_ROOT, 'f2838x'),
        'F28002': ('get_memory_c28', constants.C2000_CG_ROOT, 'f28002x'),
        'F28003': ('get_memory_c28', constants.C2000_CG_ROOT, 'f28003x'),
        'F28004': ('get_memory_c28', constants.C2000_CG_ROOT, 'f28004x'),
        'F280013': ('get_memory_c28', constants.C2000_CG_ROOT, 'f280013x'),
        'F280015': ('get_memory_c28', constants.C2000_CG_ROOT, 'f280015x'),
        'F29H85': ('get_memory_c29', constants.CG_TOOL_ROOT, 'f29h85x'),
    }
    return devices.get(device_name)


def _resolve_input_features(fe_params, override=None):
    """Resolve the model's input feature-dimension from feature-extraction
    preset/config values, following the same convention encoded in this
    codebase's own feature-extraction preset names
    (e.g. 'FFT1024Input_256Feature_1Frame' -> 256*1 = 256 model input
    features from a 1024-sample raw frame): when binning/feature-extraction
    reduces a frame to `feature_size_per_frame` features repeated over
    `num_frame_concat` frames, the model sees `feature_size_per_frame *
    num_frame_concat` features; otherwise (no reduction, e.g. RAW_FE/
    ROUND_OFF/passthrough transforms) the model sees the raw `frame_size`.
    """
    if override is not None:
        return int(override), 'override'
    feature_size_per_frame = fe_params.get('feature_size_per_frame')
    num_frame_concat = fe_params.get('num_frame_concat') or 1
    frame_size = fe_params.get('frame_size') or 1
    if feature_size_per_frame:
        return int(feature_size_per_frame) * int(num_frame_concat), 'config (feature_size_per_frame * num_frame_concat)'
    return int(frame_size), 'config (frame_size)'


def _count_cached_classes(dataset_path):
    """Best-effort: if this dataset has already been downloaded/prepared by
    a previous real run, ModelMaker leaves a stable local cache at
    <dataset_path>/classes/<class_name>/ (and, before the first run's split,
    <dataset_path>/downloaded_dataset/classes/<class_name>/). Count those
    subdirectories to get num_classes without touching the network or
    re-running dataset preparation. Returns None if no such cache exists.
    """
    if not dataset_path:
        return None
    for candidate in (os.path.join(dataset_path, 'classes'),
                       os.path.join(dataset_path, 'downloaded_dataset', 'classes')):
        if os.path.isdir(candidate):
            class_dirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
            if class_dirs:
                return len(class_dirs)
    return None


def _find_cached_user_input_config(project_path):
    """Best-effort: reuse a golden_vectors/user_input_config.h left behind by
    a previous real training run of this same project (any run timestamp),
    so the FEL half of the memory report has a chance to succeed even though
    this standalone check never runs real feature extraction. Returns the
    path to the most recently modified match, or None if none exists.
    """
    if not project_path:
        return None
    run_dir = os.path.join(project_path, 'run')
    if not os.path.isdir(run_dir):
        return None
    candidates = []
    for root, _dirs, files in os.walk(run_dir):
        if os.path.basename(root) == 'golden_vectors' and 'user_input_config.h' in files:
            candidates.append(os.path.join(root, 'user_input_config.h'))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Estimate the deployed ROM/RAM footprint of the model architecture described '
                     'by a ModelMaker config, without downloading a dataset, training, or running '
                     'the full compilation pipeline.')
    parser.add_argument('config_file', type=str, help='path to a ModelMaker YAML (or JSON) config file')
    parser.add_argument('--target_device', type=str, default=None, help='override common.target_device')
    parser.add_argument('--model_name', type=str, default=None, help='override training.model_name')
    parser.add_argument('--variables', type=int, default=None,
                         help='override the number of input variables/channels (skips config resolution)')
    parser.add_argument('--input-features', type=int, default=None,
                         help='override the model input feature dimension (skips config resolution)')
    parser.add_argument('--num-classes', type=int, default=None,
                         help='override num_classes / model output size (skips cached-dataset resolution)')
    parser.add_argument('--opset-version', type=int, default=18, help='ONNX opset used for the export (default: 18, matching train_base.py)')
    parser.add_argument('--quantization', type=int, default=None, choices=[0, 1, 2],
                         help='TinyMLQuantizationVersion to also estimate alongside the float32 baseline: '
                              '0=NO_QUANTIZATION, 1=QUANTIZATION_GENERIC, 2=QUANTIZATION_TINPU (matches '
                              "train_base.py's --quantization flag -- this is a quantization SCHEME, not a "
                              'bitwidth; use --weight-bitwidth/--activation-bitwidth for that). Default: the '
                              "config's own resolved training.quantization value (usually 2/TINPU); pass "
                              '--quantization 0 to force float32-only.')
    parser.add_argument('--quantization-method', type=str, default='QAT', choices=['QAT', 'PTQ'],
                         help='quantization flavour used when --quantization is non-zero (default: QAT, '
                              'matching train_base.py)')
    parser.add_argument('--weight-bitwidth', type=int, default=8, choices=[16, 8, 4, 2],
                         help='weight bitwidth used when --quantization is non-zero (default: 8)')
    parser.add_argument('--activation-bitwidth', type=int, default=8,
                         help='activation bitwidth used when --quantization is non-zero (default: 8)')
    parser.add_argument('--skip-fel', action='store_true',
                         help="don't attempt the Feature Extraction Library memory estimate, only the NN model")
    parser.add_argument('--output-dir', type=str, default=None,
                         help='throwaway directory for the preflight export/compile artifacts '
                              '(default: <project_run_path>/memory_estimate_only, or a temp dir)')
    return parser


def main(args):
    with open(args.config_file) as fp:
        if args.config_file.endswith('.json'):
            import json
            config = json.load(fp)
        else:
            config = yaml.safe_load(fp)

    config.setdefault('common', {})
    config.setdefault('dataset', {})
    config.setdefault('data_processing_feature_extraction', {})
    config.setdefault('training', {})
    config.setdefault('compilation', {})

    if args.target_device:
        config['common']['target_device'] = args.target_device
    if args.model_name:
        config['training']['model_name'] = args.model_name

    target_device = config['common']['target_device']
    task_type = config['common']['task_type']
    config['common']['task_type'] = task_type

    import tinyml_modelmaker
    task_category = tinyml_modelmaker.get_task_category_type_from_task_type(task_type)
    config['common']['task_category'] = task_category

    if 'target_module' in config['common']:
        target_module = config['common']['target_module']
    else:
        target_module = tinyml_modelmaker.get_target_module_from_task_type(task_type)
        if target_module is None:
            logger.error(f"Could not infer target_module from task_type '{task_type}'. "
                         f"Please specify 'target_module' in config.")
            return 1
        config['common']['target_module'] = target_module

    if target_module != 'timeseries':
        logger.error(f"estimate_memory.py currently only supports target_module='timeseries' "
                     f"(memory pre-flight/FEL wiring for '{target_module}' is not available). "
                     f"Got task_type='{task_type}' -> target_module='{target_module}'.")
        return 1

    ai_target_module = tinyml_modelmaker.ai_modules.get_target_module(target_module)

    model_name = config['training']['model_name']
    nas_enabled = config.get('training', {}).get('nas_enabled', False)
    if nas_enabled:
        logger.error("Memory pre-flight is not supported for NAS-enabled configs (training.nas_enabled=True): "
                     "NAS produces its architecture through a search procedure, so there is no fixed "
                     "architecture to instantiate ahead of time.")
        return 1

    params = ai_target_module.runner.ModelRunner.init_params()
    model_description = ai_target_module.runner.ModelRunner.get_model_description(model_name)
    if model_description is None:
        logger.error(f"please check if the given model_name is a supported one: {model_name}")
        return 1

    dataset_preset_descriptions = ai_target_module.runner.ModelRunner.get_dataset_preset_descriptions(params)
    dataset_preset_name = ai_target_module.constants.DATASET_DEFAULT
    if 'dataset_name' in config['dataset']:
        dataset_preset_name = config['dataset']['dataset_name']
    dataset_preset_description = dataset_preset_descriptions.get(dataset_preset_name) or dict()

    feature_extraction_preset_descriptions = ai_target_module.runner.ModelRunner.get_feature_extraction_preset_descriptions(params)
    feature_extraction_preset_name = ai_target_module.constants.FEATURE_EXTRACTION_DEFAULT
    if 'feature_extraction_name' in config['data_processing_feature_extraction']:
        feature_extraction_preset_name = config['data_processing_feature_extraction']['feature_extraction_name']
    feature_extraction_preset_description = feature_extraction_preset_descriptions.get(feature_extraction_preset_name) or dict()

    preset_descriptions = ai_target_module.runner.ModelRunner.get_preset_descriptions(params)
    compilation_preset_name = ai_target_module.constants.COMPILATION_DEFAULT
    if 'compile_preset_name' in config['compilation']:
        compilation_preset_name = config['compilation']['compile_preset_name']
    if target_device not in preset_descriptions:
        logger.error(f"target_device '{target_device}' is not supported. "
                     f"Supported devices: {sorted(preset_descriptions.keys())}")
        return 1
    if task_type not in preset_descriptions[target_device]:
        logger.error(f"task_type '{task_type}' is not supported for device '{target_device}'. "
                     f"Supported task types for this device: {sorted(preset_descriptions[target_device].keys())}")
        return 1
    if compilation_preset_name not in preset_descriptions[target_device][task_type].keys():
        logger.warning(f'Using "default_preset" for compilation since user choice-"{compilation_preset_name}" is unavailable')
        compilation_preset_name = 'default_preset'
    compilation_preset_description = preset_descriptions[target_device][task_type][compilation_preset_name]

    params = params.update(model_description or {}).update(dataset_preset_description) \
        .update(feature_extraction_preset_description).update(compilation_preset_description).update(config)

    # ModelRunner.__init__ only resolves/normalizes paths (project_path, dataset_path,
    # project_run_path, ...) and auto-detects data_dir -- no download, no dataset load,
    # no training happens here. verbose=False keeps it from dumping the entire params
    # tree to the log for what is meant to be a quick check.
    try:
        model_runner = ai_target_module.runner.ModelRunner(params, verbose=False)
    except Exception as exc:
        logger.error(f"Could not resolve config: {exc}")
        return 1
    params = model_runner.get_params()

    fe_params = params.data_processing_feature_extraction

    variables = args.variables if args.variables is not None else fe_params.get('variables', 1)
    variables_source = 'override' if args.variables is not None else 'config (data_processing_feature_extraction.variables)'

    input_features, input_features_source = _resolve_input_features(fe_params, override=args.input_features)

    num_classes = args.num_classes
    num_classes_source = 'override'
    if num_classes is None:
        if task_category == ai_target_module.constants.TASK_CATEGORY_TS_ANOMALYDETECTION:
            num_classes = input_features
            num_classes_source = 'auto (anomaly detection: num_classes == input_features, autoencoder output)'
        elif task_category == ai_target_module.constants.TASK_CATEGORY_TS_FORECASTING:
            target_variables = fe_params.get('target_variables') or []
            num_target_variables = len(target_variables) if target_variables else 1
            forecast_horizon = fe_params.get('forecast_horizon') or 1
            num_classes = int(forecast_horizon) * int(num_target_variables)
            num_classes_source = 'auto (forecasting: forecast_horizon * num_target_variables)'
        else:
            cached = _count_cached_classes(params.dataset.dataset_path)
            if cached is not None:
                num_classes = cached
                num_classes_source = f'cached dataset ({params.dataset.dataset_path}/classes)'

    if num_classes is None:
        logger.error(
            "Could not resolve num_classes from the config or from any locally cached copy of the "
            f"dataset at '{params.dataset.dataset_path}'. This dataset has apparently never been "
            "downloaded/prepared by a previous run, and num_classes cannot be known without either "
            "the real dataset or an explicit value. Re-run with --num-classes N (this CLI will not "
            "trigger a dataset download just to find out).")
        return 1

    model_name_display = params.training.model_name or model_name
    print(f"Estimating memory for: {args.config_file} (model: {model_name_display}, device: {target_device})")
    print(f"  variables={variables} [{variables_source}]")
    print(f"  input_features={input_features} [{input_features_source}]")
    print(f"  num_classes={num_classes} [{num_classes_source}]")

    try:
        from tinyml_tinyverse.common import models
    except ImportError as exc:
        logger.error(f"Could not import tinyml_tinyverse model registry: {exc}")
        return 1

    try:
        model = models.get_model(
            params.training.model_training_id, variables, num_classes,
            input_features=input_features, model_config=params.training.model_config,
            model_spec=params.training.model_spec, dual_op=params.training.dual_op)
    except Exception as exc:
        logger.error(f"Could not instantiate model '{params.training.model_training_id}': {exc}")
        return 1

    fel_config = None if args.skip_fel else _get_device_fel_function(ai_target_module, target_device)
    if not args.skip_fel and fel_config is None:
        logger.info(f"No Feature Extraction Library memory mapping for device '{target_device}' "
                    f"(this is a normal/valid case for some devices) -- only the NN model estimate will run.")

    if args.output_dir:
        output_dir = args.output_dir
    elif params.common.project_run_path:
        output_dir = os.path.join(params.common.project_run_path, 'memory_estimate_only')
    else:
        output_dir = tempfile.mkdtemp(prefix='tinyml_memory_estimate_')
    os.makedirs(output_dir, exist_ok=True)

    if fel_config is not None:
        cached_config_h = _find_cached_user_input_config(params.common.project_path)
        if cached_config_h:
            dest_dir = os.path.join(output_dir, 'training', 'quantization', 'golden_vectors')
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copyfile(cached_config_h, os.path.join(dest_dir, 'user_input_config.h'))
            logger.info(f"Reusing cached feature-extraction header for the FEL estimate: {cached_config_h}")

    try:
        from tinyml_tinyverse.references.common.memory_preflight import estimate_memory
    except ImportError as exc:
        logger.error(f"tinyml_tinyverse does not provide memory_preflight.estimate_memory yet: {exc}")
        return 1

    # Some model families (e.g. NPU-targeted conv2d models) expect a 4D input
    # with a trailing singleton dim -- see train_base.py's log_model_summary,
    # which uses (1, variables, input_features, 1) for exactly this reason.
    # We have no real dataset here to know which convention this model needs
    # (that's the whole point of this standalone check), so try the plain 3D
    # shape first and fall back to the 4D one if the model rejects it.
    candidate_shapes = [(1, variables, input_features), (1, variables, input_features, 1)]

    # --quantization is a TinyMLQuantizationVersion (0/1/2), a quantization SCHEME,
    # not a bitwidth -- default to the config's own resolved training.quantization
    # (usually 2/QUANTIZATION_TINPU) unless the user overrode it.
    quant_version = args.quantization if args.quantization is not None else int(params.training.quantization)
    quant_label = f"{args.weight_bitwidth}-bit weight / {args.activation_bitwidth}-bit activation"

    if quant_version == 0:
        quantization_values = [0]  # Only base (float32) estimate
        print(f"\nRunning memory pre-flight for base (float32) model...")
    else:
        quantization_values = [0, quant_version]  # Both base and quantized estimates
        print(f"\nRunning memory pre-flight for base (float32) and quantized "
              f"(scheme={quant_version}, {quant_label}, method={args.quantization_method}) models...")

    results = {}  # Store results for each quantization value

    for quant_val in quantization_values:
        print(f"  Testing quantization={quant_val}...")
        result = None
        for shape in candidate_shapes:
            result = estimate_memory(
                model, input_shape=shape, output_dir=output_dir,
                target=params.compilation.target, cross_compiler=params.compilation.cross_compiler,
                cross_compiler_options=params.compilation.cross_compiler_options,
                target_c_mcpu=params.compilation.target_c_mcpu, opset_version=args.opset_version,
                quantization=quant_val, quantization_method=args.quantization_method,
                weight_bitwidth=args.weight_bitwidth, activation_bitwidth=args.activation_bitwidth,
                fel_config=fel_config, logger=logger)
            if result.get('model') is not None:
                break

        if result.get('model') is None:
            logger.error(f"Memory pre-flight failed to produce a NN model memory report for quantization={quant_val} "
                         f"after trying input shapes {candidate_shapes}. "
                         f"See {os.path.join(output_dir, 'memory_preflight_full.log')} for details.")
            return 1

        results[quant_val] = result

        if fel_config is not None and result.get('fel') is None:
            logger.warning(f"NN model memory report succeeded for quantization={quant_val}, but the FEL memory report did not "
                           "(see above for the reason). Continuing with exit code 0.")

    # Print formatted results
    print("\n" + "="*60)
    print("MEMORY PRE-FLIGHT RESULTS")
    print("="*60)

    def _print_block(label, info):
        print(f"\n{label}:")
        print(f"  Code: {info['code']} bytes ({info['code']/1024:.2f} KB)")
        print(f"  RO Data: {info['ro_data']} bytes ({info['ro_data']/1024:.2f} KB)")
        print(f"  RW Data: {info['rw_data']} bytes ({info['rw_data']/1024:.2f} KB)")
        print(f"  Total: {info['total']} bytes ({info['total']/1024:.2f} KB)")

    # Print base (float32) results
    base_result = results[0]
    if base_result.get('model'):
        _print_block("Base Model (float32)", base_result['model'])
    # FEL's compiled artifacts depend on the paired NN model's own compiled
    # I/O format (skip_normalize/output_int differ float32 vs. quantized), so
    # print it once per model variant rather than a single shared number.
    if fel_config is not None:
        if base_result.get('fel'):
            _print_block("Feature Extraction Library (FEL, paired with float32 model)", base_result['fel'])
        else:
            print(f"\nFeature Extraction Library (FEL, paired with float32 model): Not available for this device")

    if quant_version > 0 and 0 in results and quant_version in results:
        # Print quantized results
        quant_result = results[quant_version]
        if quant_result.get('model'):
            _print_block(f"Quantized Model ({quant_label}, scheme={quant_version})", quant_result['model'])

            # Calculate savings
            base_total = base_result['model']['total']
            quant_total = quant_result['model']['total']
            if base_total > 0:
                savings = ((base_total - quant_total) / base_total) * 100
                print(f"\nMemory Savings: {savings:.1f}% reduction")

        if fel_config is not None:
            if quant_result.get('fel'):
                _print_block(f"Feature Extraction Library (FEL, paired with quantized model)", quant_result['fel'])
            else:
                print(f"\nFeature Extraction Library (FEL, paired with quantized model): Not available for this device")

    print("="*60)

    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s: %(message)s')
    # the cwd must be the root of the repository, matching run_tinyml_modelmaker.py
    if os.path.split(os.getcwd())[-1] == 'tinyml_modelmaker':
        os.chdir('..')
    #
    parsed_args = get_args_parser().parse_args()
    sys.exit(main(parsed_args))
