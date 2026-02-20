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
#
# Few lines are from: https://github.com/pytorch/vision
# BSD 3-Clause License - Copyright (c) Soumith Chintala 2016
#################################################################################

"""
Time series anomaly detection training script.
"""

import os
import random
import sys
import timeit
from logging import getLogger

import numpy as np
import onnxruntime as ort
from tinyml_torchmodelopt.quantization import TinyMLQuantizationMethod

import torch
import torch.nn as nn

from tinyml_tinyverse.common import models
from tinyml_tinyverse.common.datasets import GenericTSDatasetAD
from tinyml_tinyverse.common.utils import misc_utils, utils, ondevice_training

# Import common functions from base module
from ..common.train_base import (
    get_base_args_parser,
    generate_golden_vector_dir,
    generate_test_vector,
    load_datasets,
    run_distributed,
    assemble_golden_vectors_header,
    setup_training_environment,
    prepare_transforms,
    create_model,
    log_model_summary,
    load_pretrained_weights,
    setup_optimizer_and_scheduler,
    setup_distributed_model,
    resume_from_checkpoint,
    save_checkpoint,
    handle_export_only,
    move_model_to_device,
    log_training_time,
    apply_output_int_default,
    get_output_int_flag,
    load_onnx_for_inference,
    create_data_loaders,
)

dataset_loader_dict = {'GenericTSDatasetAD': GenericTSDatasetAD}
dataset_load_state = {'dataset': None, 'dataset_test': None, 'train_sampler': None, 'test_sampler': None}


def get_args_parser():
    """Create argument parser with anomaly detection-specific arguments."""
    parser = get_base_args_parser("This script loads time series data and trains an anomaly detection model")

    # Override defaults for anomaly detection
    for action in parser._actions:
        if action.dest == 'loader_type':
            action.default = 'anomalydetection'
        elif action.dest == 'opt':
            action.default = 'Adam'
        elif action.dest == 'lr':
            action.default = 0.001

    return parser


def generate_user_input_config_ad(output_dir, dataset, threshold):
    """Generate user_input_config.h with anomaly detection threshold."""
    logger = getLogger("root.generate_user_input_config")
    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    user_input_config_h = os.path.join(golden_vectors_dir, 'user_input_config.h')
    logger.info("Creating user_input_config.h at: {}".format(user_input_config_h))

    with open(user_input_config_h, 'w') as fp:
        fp.write("#ifndef INPUT_CONFIG_H_\n")
        fp.write("#define INPUT_CONFIG_H_\n\n")
        fp.write(''.join([f'#define {flag}\n' for flag in dataset.preprocessing_flags]))
        fp.write('\n'.join([f'#define {k} {v}' for k, v in dataset.feature_extraction_params.items()]))
        fp.write(f'\n#define RECONSTRUCTION_ERROR_THRESHOLD {threshold}\n')
        fp.write("\n\n#endif /* INPUT_CONFIG_H_ */\n")


def generate_golden_vectors(output_dir, dataset, output_int, threshold, generic_model=False):
    """Generate golden vectors for anomaly detection."""
    logger = getLogger("root.generate_golden_vectors")
    ort_sess, input_name, output_name = load_onnx_for_inference(output_dir, generic_model)
    vector_files = []

    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    logger.info(f"Creating Golden data for reference at {golden_vectors_dir}")

    for index in random.sample(list(range(len(dataset.X))), k=4):
        np_raw = dataset.X_raw[index]
        np_feat = dataset.X[index]
        pred = ort_sess.run([output_name], {input_name: np.expand_dims(np_feat, 0).astype(np.float32)})[0]
        half_path = os.path.join(golden_vectors_dir)

        np.savetxt(half_path + f'adc_{index}.txt', np_raw.flatten(),
                   fmt='%f,' if np_raw.dtype.kind == 'f' else '%d,',
                   header=f'//(Index: {index}): ADC Data\nfloat raw_input_test[{len(np_raw.flatten())}]= {{',
                   footer='}', comments='', newline=' ')
        vector_files.append(half_path + f'adc_{index}.txt')
        np.savetxt(half_path + f'features_{index}.txt', np_feat.flatten(),
                   fmt='%f,' if np_feat.dtype.kind == 'f' else '%d,',
                   header=f'//(Index: {index}): Extracted Features\nfloat model_test_input[{len(np_feat.flatten())}] = {{',
                   footer='}', comments='', newline=' ')
        vector_files.append(half_path + f'features_{index}.txt')
        np.savetxt(half_path + f'output_{index}.txt', pred.flatten(),
                   fmt='%d,' if output_int else '%f,',
                   header=f'//(Index: {index}):  Expected Model Output\n{"int8_t" if output_int else "float"} golden_output[{len(pred.flatten())}] = {{',
                   footer='}', comments='', newline=' ')
        vector_files.append(half_path + f'output_{index}.txt')

    header_file_info = assemble_golden_vectors_header(vector_files, files_per_set=3)
    generate_test_vector(output_dir, header_file_info)
    generate_user_input_config_ad(output_dir, dataset, threshold)


def get_reconstruction_errors_stats(generic_model, model_path, device, data_loader):
    """Calculate reconstruction error statistics for threshold determination."""
    if not generic_model:
        utils.decrypt(model_path, utils.get_crypt_key())
    ort_sess = ort.InferenceSession(model_path)
    if not generic_model:
        utils.encrypt(model_path, utils.get_crypt_key())

    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name
    errors = torch.tensor([]).to(device, non_blocking=True)
    for _, data, targets in data_loader:
        data = data.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long()
        batch_reconstruction_errors = torch.tensor([]).to(device, non_blocking=True)
        for input, target_label in zip(data, targets):
            input = input.unsqueeze(0).cpu().numpy()
            output = torch.tensor(ort_sess.run([output_name], {input_name: input})[0]).to(device)
            current_output_error = torch.mean((torch.from_numpy(input).to(device) - output) ** 2, dim=(1, 2, 3))
            batch_reconstruction_errors = torch.cat((batch_reconstruction_errors, current_output_error))
        errors = torch.cat((errors, batch_reconstruction_errors))

    normal_error_mean = torch.mean(errors)
    normal_error_std = torch.std(errors)
    return normal_error_mean.cpu(), normal_error_std.cpu()


def main(gpu, args):
    """Main training function for anomaly detection."""
    logger, device = setup_training_environment(args, gpu, 'anomalydetection', __file__)
    prepare_transforms(args)

    # Load or reuse datasets
    if args.quantization:
        dataset, dataset_test, train_sampler, test_sampler = (dataset_load_state['dataset'], dataset_load_state['dataset_test'],
                                                               dataset_load_state['train_sampler'], dataset_load_state['test_sampler'])
    else:
        dataset, dataset_test, train_sampler, test_sampler = load_datasets(args.data_path, args, dataset_loader_dict)
        dataset_load_state['dataset'], dataset_load_state['dataset_test'] = dataset, dataset_test
        dataset_load_state['train_sampler'], dataset_load_state['test_sampler'] = train_sampler, test_sampler

    num_classes = len(dataset.classes)
    variables = dataset.X.shape[1]
    input_features = dataset.X.shape[2]

    logger.info("Loading data:")
    data_loader, data_loader_test = create_data_loaders(dataset, dataset_test, train_sampler, test_sampler, args, gpu)

    logger.info("Creating model")
    logger.info(f"Variables: {variables}, Input_features: {input_features}")

    # For anomaly detection, num_classes is input_features (autoencoder output)
    model = models.get_model(
        args.model, variables, num_classes=input_features, input_features=input_features, model_config=args.model_config,
        model_spec=args.model_spec,
        dual_op=args.dual_op)

    log_model_summary(model, args, variables, input_features, logger)
    model = load_pretrained_weights(model, args, logger)

    # if output_int not set by user, then set it to default of task_type
    if args.output_int == None:
        args.output_int = False
    model = utils.quantization_wrapped_model(
        model, args.quantization, args.quantization_method, args.weight_bitwidth, args.activation_bitwidth,
        args.epochs, args.output_int)
    

    if handle_export_only(model, args, variables, input_features, logger):
        return

    move_model_to_device(model, device, logger)
    criterion = nn.MSELoss()

    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, args)
    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
    resume_from_checkpoint(model_without_ddp, optimizer, lr_scheduler, model_ema, args)

    phase = 'QuantTrain' if args.quantization else 'FloatTrain'
    logger.info("Start training")
    start_time = timeit.default_timer()
    best = dict(mse=np.inf, epoch=None)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        utils.train_one_epoch_anomalydetection(
            model, criterion, optimizer, data_loader, device, epoch, None, args.apex, model_ema,
            print_freq=args.print_freq, phase=phase, num_classes=num_classes, dual_op=args.dual_op,
            is_ptq=True if (args.quantization_method in ['PTQ'] and args.quantization) else False)
        if not (args.quantization_method in ['PTQ'] and args.quantization):
            lr_scheduler.step()
        avg_mse = utils.evaluate_anomalydetection(model, criterion, data_loader_test, device=device,
                                                   transform=None, print_freq=args.print_freq, epoch=epoch,
                                                   phase=phase, num_classes=num_classes, dual_op=args.dual_op)
        if model_ema:
            avg_mse = utils.evaluate_anomalydetection(
                model_ema, criterion, data_loader_test, device=device, transform=None, epoch=epoch,
                log_suffix='EMA', print_freq=args.print_freq, phase=phase, dual_op=args.dual_op)
        if args.output_dir and avg_mse <= best['mse']:
            logger.info(f"Epoch {epoch}: {avg_mse:.2f} (Val MSE) <= {best['mse']:.2f} (So far least error). Hence updating checkpoint.pth")
            best['mse'], best['epoch'] = avg_mse, epoch
            checkpoint = save_checkpoint(model_without_ddp, optimizer, lr_scheduler, epoch, args, model_ema)
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    # Log best epoch results
    logger = getLogger(f"root.main.{phase}.BestEpoch")
    logger.info("")
    logger.info("Printing statistics of best epoch:")
    logger.info(f"Best Epoch: {best['epoch']}")
    logger.info(f"MSE {best['mse']:.3f}")
    logger.info("")

    # Export model
    logger.info('Exporting model after training.')
    if args.distributed is False or (args.distributed is True and int(os.environ['LOCAL_RANK']) == 0):
        utils.export_model(
            model, input_shape=(1,) + dataset.X.shape[1:], output_dir=args.output_dir, opset_version=args.opset_version,
            quantization=args.quantization, example_input=None, generic_model=args.generic_model,
            remove_hooks_for_jit=True if (args.quantization_method == TinyMLQuantizationMethod.PTQ and args.quantization) else False)
        if args.ondevice_training:
            saved_onnx_path = os.path.join(args.output_dir, 'model.onnx')
            ondevice_training.export_for_ondevice_training(saved_onnx_path, args)

    log_training_time(start_time)

    # Calculate threshold
    model_path = os.path.join(args.output_dir, 'model.onnx')
    error_mean, error_std = get_reconstruction_errors_stats(args.generic_model, model_path, args.device, data_loader)
    threshold = error_mean + 3 * error_std

    if args.gen_golden_vectors:
        generate_golden_vector_dir(args.output_dir)
        output_int = get_output_int_flag(args)
        generate_golden_vectors(args.output_dir, dataset, output_int, threshold, args.generic_model)


def run(args):
    """Run training with optional distributed mode."""
    run_distributed(main, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    # Apply default output_int if not specified by user
    apply_output_int_default(arguments, 'timeseries_anomalydetection')
    run(arguments)
