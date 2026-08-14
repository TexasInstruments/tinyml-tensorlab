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
Radar classification training script.
"""

import os
import random
import sys
import timeit
from argparse import Namespace
from logging import getLogger

import numpy as np
import pandas as pd
from tabulate import tabulate

from tinyml_tinyverse.common.models import NeuralNetworkWithPreprocess
from torcheval.metrics.functional import multiclass_confusion_matrix, multiclass_f1_score, multiclass_auroc, r2_score, mean_squared_error
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion, TinyMLQuantizationMethod
from tinyml_torchmodelopt.nas.train_cnn_search import search_and_get_model

import torch
import torch.nn as nn
import torchinfo

from tinyml_tinyverse.common import models
from tinyml_tinyverse.common.datasets import GenericRadarDataset
from tinyml_tinyverse.common.utils import misc_utils, utils, gof_utils
from tinyml_tinyverse.common.utils.mdcl_utils import Logger

# Import common functions from base module
from ..common.train_base import (
    get_base_args_parser,
    generate_golden_vector_dir,
    generate_user_input_config,
    generate_test_vector,
    generate_model_aux,
    load_datasets,
    run_distributed,
    assemble_golden_vectors_header,
    setup_training_environment,
    prepare_transforms,
    create_data_loaders,
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
)

dataset_loader_dict = {'GenericRadarDataset': GenericRadarDataset}
dataset_load_state = {'dataset': None, 'dataset_test': None, 'train_sampler': None, 'test_sampler': None}


def get_args_parser():
    """Create argument parser with classification-specific arguments."""
    DESCRIPTION = "This script loads radar data and trains it generating a model"
    parser = get_base_args_parser("This script loads radar data and trains a radar classification model")

    # Classification-specific arguments
    parser.add_argument('--gof-test', type=misc_utils.str2bool, default=False, help='Enable goodness-of-fit test')
    parser.add_argument('--file-level-classification-log', help='File level classification log file', type=str)

    # Radar Detection related params
    #parser.add_argument('--frame-size', default=8, type=int, help='Number of frames per window ')

    # NAS arguments
    parser.add_argument("--nas_enabled", default=False, help="Enable/ Disable NAS", type=misc_utils.str2bool)
    parser.add_argument("--nas_optimization_mode", default="Memory", type=str, help="Optimize model for compute or storage efficiency")
    parser.add_argument("--nas_model_size", default='None', choices=['s', 'm', 'l', 'xl', 'None'], help="Proxy for model size")
    parser.add_argument("--nas_epochs", default=10, type=int, help="Iterations for search")
    parser.add_argument("--nas_nodes_per_layer", default=4, type=int, help="Number of nodes per layer")
    parser.add_argument("--nas_layers", default=3, type=int, help="Should be minimum 3")
    parser.add_argument("--nas_init_channels", default=1, type=int, help="Initial channel size of the first feature map")
    parser.add_argument("--nas_init_channel_multiplier", default=3, type=int, help="Channel size of after first preprocess")
    parser.add_argument("--nas_fanout_concat", default=4, type=int, help="Number of nodes to concat for output after each layer")
    parser.add_argument("--load_saved_model", type=str, default='None', help="Model path for pre-searched nas model")

    return parser


def get_nas_args(args, data_loader, data_loader_test, num_classes, variables):
    """Configure NAS arguments based on model size preset."""
    if args.nas_model_size != "None":
        model_size = args.nas_model_size
        if model_size == 's':
            args.nas_nodes_per_layer, args.nas_layers = 4, 3
            args.nas_init_channels, args.nas_init_channel_multiplier, args.nas_fanout_concat = 1, 3, 4
        elif model_size == 'm':
            args.nas_nodes_per_layer, args.nas_layers = 4, 10
            args.nas_init_channels, args.nas_init_channel_multiplier, args.nas_fanout_concat = 1, 3, 4
        elif model_size == 'l':
            args.nas_nodes_per_layer, args.nas_layers = 4, 12
            args.nas_init_channels, args.nas_init_channel_multiplier, args.nas_fanout_concat = 4, 3, 4
        elif model_size == 'xl':
            args.nas_nodes_per_layer, args.nas_layers = 4, 20
            args.nas_init_channels, args.nas_init_channel_multiplier, args.nas_fanout_concat = 4, 3, 4
        elif model_size == 'xxl':
            args.nas_nodes_per_layer, args.nas_layers = 6, 20
            args.nas_init_channels, args.nas_init_channel_multiplier, args.nas_fanout_concat = 8, 3, 4

    nas_args_dict = {
        'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay, 'gpu': 0,
        'nas_budget': args.nas_epochs, 'nas_init_channels': args.nas_init_channels,
        'nas_nodes_per_layer': args.nas_nodes_per_layer, 'nas_layers': args.nas_layers,
        'nas_multiplier': args.nas_fanout_concat, 'nas_stem_multiplier': args.nas_init_channel_multiplier,
        'nas_optimization_mode': args.nas_optimization_mode, 'in_channels': variables, 'grad_clip': 5,
        'mode': 'cnn', 'arch_learning_rate': 1e-2, 'arch_weight_decay': 1e-3, 'unrolled': True,
        'num_classes': num_classes, 'train_loader': data_loader, 'valid_loader': data_loader_test,
    }
    return Namespace(**nas_args_dict)


def generate_golden_vectors(output_dir, dataset, output_int, generic_model=False):
    """Generate golden vectors for radar classification."""
    logger = getLogger("root.generate_golden_vectors")
    ort_sess, input_name, output_name = load_onnx_for_inference(output_dir, generic_model)
    vector_files = []

    golden_vectors_dir = os.path.join(output_dir, 'golden_vectors')
    logger.info(f"Creating Golden data for reference at {golden_vectors_dir}")
    label_index_dict = {dataset.inverse_label_map.get(label): np.where(dataset.Y == label)[0] for label in np.unique(dataset.Y)}

    for label, indices in label_index_dict.items():
        for index in random.sample(list(indices), k=2):
            np_feat = np.array(dataset.X[index], dtype=np.float32)
            pred = ort_sess.run([output_name], {input_name: np.expand_dims(np_feat, 0)})[0]

            half_path = os.path.join(golden_vectors_dir)

            np.savetxt(half_path + f'features_{label}_{index}.txt', np_feat.flatten(), fmt='%.5f,',
                       header=f'//Class: {label} (Index: {index}): Feature Data\nfloat model_test_input[{len(np_feat.flatten())}] = {{',
                       footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'features_{label}_{index}.txt')
            np.savetxt(half_path + f'output_{label}_{index}.txt', pred.flatten(),
                       fmt='%d,' if output_int else '%f,',
                       header=f'//Class: {label} (Index: {index}): Expected Model Output\n{"int8_t" if output_int else "float"} golden_output[{len(pred.flatten())}] = {{',
                       footer='}', comments='', newline=' ')
            vector_files.append(half_path + f'output_{label}_{index}.txt')

    header_file_info = assemble_golden_vectors_header(vector_files, files_per_set=2)
    generate_user_input_config(output_dir, dataset)
    generate_test_vector(output_dir, header_file_info)
    generate_model_aux(output_dir, dataset)


def main(gpu, args):
    """Main training function for classification."""
    logger, device = setup_training_environment(args, gpu, 'classification', __file__)
    prepare_transforms(args)

    # Load or reuse datasets
    if args.quantization:
        dataset, dataset_test, train_sampler, test_sampler = (dataset_load_state['dataset'], dataset_load_state['dataset_test'],
                                                               dataset_load_state['train_sampler'], dataset_load_state['test_sampler'])
    else:
        dataset, dataset_test, train_sampler, test_sampler = load_datasets(args.data_path, args, dataset_loader_dict)
        dataset_load_state['dataset'], dataset_load_state['dataset_test'] = dataset, dataset_test
        dataset_load_state['train_sampler'], dataset_load_state['test_sampler'] = train_sampler, test_sampler

        try:
            utils.plot_feature_components_graph(dataset, graph_type='pca', instance_type='train', output_dir=args.output_dir)
            utils.plot_feature_components_graph(dataset_test, graph_type='pca', instance_type='validation', output_dir=args.output_dir)
            if args.gof_test:
                if args.frame_size != 'None':
                    gof_utils.goodness_of_fit_test(frame_size=int(args.frame_size), classes_dir=args.data_path,
                                                   output_dir=args.output_dir, class_names=dataset.classes)
                else:
                    logger.warning(f"Goodness of Fit plots will not be generated because frame_size was not given in the YAML file.")
        except Exception as e:
            logger.warning(f"Feature Extraction plots will not be generated because: {e}")

    if misc_utils.str2bool(args.dont_train_just_feat_ext):
        logger.info('Exiting execution without training')
        sys.exit(0)

    num_classes = len(dataset.classes)
    variables = 1
    input_features = dataset.X.shape[1]

    logger.info("Loading data:")
    data_loader, data_loader_test = create_data_loaders(dataset, dataset_test, train_sampler, test_sampler, args, gpu)

    logger.info("Creating model")
    if args.load_saved_model == 'None':
        if args.nas_enabled == 'True':
            if args.quantization:
                model = torch.load(os.path.join(os.path.dirname(args.output_dir), os.path.join('base', 'nas_model.pt')), weights_only=False)
            else:
                nas_args = get_nas_args(args, data_loader, data_loader_test, num_classes, variables)
                model = search_and_get_model(nas_args)
                if not model:
                    logger.error("Please check on prior errors. NAS wasn't able to create a model")
                    sys.exit(1)
                torch.save(model, os.path.join(args.output_dir, 'nas_model.pt'))
        else:
            model = models.get_model(
                args.model, variables, num_classes, input_features=input_features, model_config=args.model_config,
                model_spec=args.model_spec,
                dual_op=args.dual_op)
    else:
        model = torch.load(args.load_saved_model, weights_only=False)

    if args.generic_model or args.nas_enabled:
        log_model_summary(model, args, variables, input_features, logger)

    model = load_pretrained_weights(model, args, logger)

    if handle_export_only(model, args, variables, input_features, logger):
        return

    move_model_to_device(model, device, logger)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, args)
    resume_from_checkpoint(model_without_ddp, optimizer, lr_scheduler, model_ema, args)

    phase = 'QuantTrain' if args.quantization else 'FloatTrain'
    logger.info("Start training")
    start_time = timeit.default_timer()
    best = dict(accuracy=0.0, f1=0, conf_matrix=dict(), epoch=None)

    model = NeuralNetworkWithPreprocess(None, model)

    # if output_int not set by user, then set it to default of task_type
    if args.output_int == None:
        args.output_int = True
    model = utils.quantization_wrapped_model(
        model, args.quantization, args.quantization_method, args.weight_bitwidth, args.activation_bitwidth,
        args.epochs, args.output_int)
    
    

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        utils.train_one_epoch_classification(
            model, criterion, optimizer, data_loader, device, epoch, None, args.apex, model_ema,
            print_freq=args.print_freq, phase=phase, num_classes=num_classes, dual_op=args.dual_op,
            is_ptq=True if (args.quantization_method in ['PTQ'] and args.quantization) else False)
        if not (args.quantization_method in ['PTQ'] and args.quantization):
            lr_scheduler.step()
        avg_accuracy, avg_f1, auc, avg_conf_matrix, predictions, ground_truth = utils.evaluate_classification(
            model, criterion, data_loader_test, device=device, transform=None, phase=phase,
            num_classes=num_classes, dual_op=args.dual_op)
        if model_ema:
            avg_accuracy, avg_f1, auc, avg_conf_matrix, predictions, ground_truth = utils.evaluate_classification(
                model_ema, criterion, data_loader_test, device=device, transform=None,
                log_suffix='EMA', print_freq=args.print_freq, phase=phase, dual_op=args.dual_op)
        if args.output_dir and avg_accuracy >= best['accuracy']:
            logger.info(f"Epoch {epoch}: {avg_accuracy:.2f} (Val accuracy) >= {best['accuracy']:.2f} (So far best accuracy). Hence updating checkpoint.pth")
            best['accuracy'], best['f1'], best['auc'], best['conf_matrix'], best['epoch'] = avg_accuracy, avg_f1, auc, avg_conf_matrix, epoch
            best['predictions'], best['ground_truth'] = predictions, ground_truth
            checkpoint = save_checkpoint(model_without_ddp, optimizer, lr_scheduler, epoch, args, model_ema)
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    # Log best epoch results
    logger = getLogger(f"root.main.{phase}.BestEpoch")
    logger.info("")
    logger.info("Printing statistics of best epoch:")
    logger.info(f"Best Epoch: {best['epoch']}")
    logger.info(f"Acc@1 {best['accuracy']:.3f}")
    logger.info(f"F1-Score {best['f1']:.3f}")
    logger.info(f"AUC ROC Score {best['f1']:.3f}")
    logger.info("")
    logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(best['conf_matrix'],
                  columns=[f"Predicted as: {x}" for x in dataset.inverse_label_map.values()],
                  index=[f"Ground Truth: {x}" for x in dataset.inverse_label_map.values()]),
                                                         headers="keys", tablefmt='grid')))

    Logger(log_file=args.file_level_classification_log, DEBUG=args.DEBUG,
           name="root.utils.print_file_level_classification_summary",
           append_log=True if args.quantization else False, console_log=False)
    getLogger("root.utils.print_file_level_classification_summary").propagate = False
    utils.print_file_level_classification_summary(dataset_test, best['predictions'], best['ground_truth'], phase)
    logger.info(f"Generated file-level classification summary in: {args.file_level_classification_log}")

    # Export model
    logger.info('Exporting model after training.')
    if args.distributed is False or (args.distributed is True and int(os.environ['LOCAL_RANK']) == 0):
        example_input = next(iter(data_loader_test))[1]
        input_shape = (1,) + dataset.X.shape[1:]
        utils.export_model(
            model, input_shape=input_shape, output_dir=args.output_dir, opset_version=args.opset_version,
            quantization=args.quantization, example_input=example_input, generic_model=args.generic_model,
            remove_hooks_for_jit=True if (args.quantization_method == TinyMLQuantizationMethod.PTQ and args.quantization) else False)

    log_training_time(start_time)

    if args.gen_golden_vectors:
        generate_golden_vector_dir(args.output_dir)
        output_int = get_output_int_flag(args)
        generate_golden_vectors(args.output_dir, dataset, output_int, args.generic_model)

def main_debug(gpu, args):
    """Main training function for classification."""
    # --------Following as close as possible steps from jupyter notebook to test if model learning plateau is coming from training loop
    #First need to load everything in
    torch.manual_seed(42)
    logger, device = setup_training_environment(args, gpu, 'classification', __file__)
    prepare_transforms(args)


    # Load or reuse datasets
    dataset, dataset_test, train_sampler, test_sampler = load_datasets(args.data_path, args, dataset_loader_dict)
    dataset_load_state['dataset'], dataset_load_state['dataset_test'] = dataset, dataset_test
    dataset_load_state['train_sampler'], dataset_load_state['test_sampler'] = train_sampler, test_sampler

    num_classes = len(dataset.classes)
    variables = 1
    input_features = dataset.X.shape[1]

    logger.info("Loading data:")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers,
        pin_memory=True if gpu > 0 else False, collate_fn=utils.collate_fn, drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers,
        pin_memory=True if gpu > 0 else False, collate_fn=utils.collate_fn, drop_last=True)

    #-------1. Define Model Size and send to CPU
    logger.info("Creating model")
    model = models.get_model(args.model, variables, num_classes, 
                             input_features=input_features, model_config=args.model_config, 
                             model_spec=args.model_spec, dual_op=args.dual_op)

    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)

    # Model_0.to(device)
    move_model_to_device(model, device, logger)
    # loss_fn in jupyter notebook
    criterion = nn.CrossEntropyLoss()

    #setup optimizer function
    optimizer = torch.optim.SGD(
                model.parameters(), lr =args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    phase = 'FloatTrain'
    logger.info("Start training")
    start_time = timeit.default_timer()
    best = dict(accuracy=0.0, f1=0, auc=0, conf_matrix=dict(), epoch=None, predictions=None, ground_truth=None)   

    #Create empty loss lists to track values
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    
    for epoch in range(args.start_epoch, args.epochs):
        #--------Training
        train_loss = 0
        header = f"Epoch: [{epoch}]"
        # Add a loop to loop through training batches
        model.train()
        for _, data, target in data_loader:
            start_time = timeit.default_timer()
            # 1. Forward pass
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)

            # 2. Calculate loss(per batch)
            loss = criterion(output, target)
            train_loss += loss.item()

            # 3. Zero gradients before forward pass
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        train_loss /= len(data_loader)


        ### Testing
        # Setup variables for accumulatively adding up loss and accuracy 
        test_loss, test_acc = 0, 0
        model.eval()
        all_preds = []
        all_labels = []
        distance = 0

        with torch.inference_mode():
            for _, data, target in data_loader_test:
                # 1. Forward pass
                data, target = data.to(device).float(), target.to(device)
                
                test_pred = model(data)

                # 2. Calculate loss (accumulatively)
                target = target.squeeze().long()
                loss = criterion(test_pred, target)
                test_loss += criterion(test_pred, target)

                # 3. Calculate accuracy y_true=y, y_pred=test_pred
                test_acc += ((test_pred.argmax(dim=1) == target).sum().item()) / len(target) * 100
                f1_score_val = multiclass_f1_score(test_pred, target, num_classes=num_classes)

                # COnvert logits to class mables
                predicted_labels = torch.argmax(test_pred, dim=1)

                # Store predictions and true labels
                all_preds.extend(predicted_labels)
                all_labels.extend(target)

                # Calculate Hamming Distance between predictions and correct categories
                a = predicted_labels.tolist()
                b = target.tolist()

                for i in range(len(a)):
                    if a[i] != b[i]:
                        distance +=1

                        #Divide total test loss by length of test data loader (per batch
            test_loss /= len(data_loader_test)
            # Divide total accuracy by length of test dataloader ( per batch)
            test_acc /= len(data_loader_test)

        
         # keep a history to view loss curves. Detach the tensors from the computation graphs.  
        epoch_count.append(epoch)
        train_loss_values.append(train_loss)
        test_loss_values.append(test_loss.detach())    

        # conf_matrix = multiclass_confusion_matrix(output, target, num_classes)
        
        avg_accuracy, avg_f1, auc, avg_conf_matrix, predictions, ground_truth = utils.evaluate_classification(
            model, criterion, data_loader_test, device=device, transform=None, phase=phase,
            num_classes=num_classes, dual_op=args.dual_op)

        ## Print out what's happening in the epoch loop
        if epoch % (args.epochs / 10) == 0 or epoch == args.epochs - 1:
            print(f"EPOCH: {epoch} | F1: {f1_score_val:.5f}")
            print(f"Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
            print(f'Distance: {distance}')

        if args.output_dir and avg_accuracy >= best['accuracy']:
            logger.info(f"Epoch {epoch}: {avg_accuracy:.2f} (Val accuracy) >= {best['accuracy']:.2f} (So far best accuracy). Hence updating checkpoint.pth")
            best['accuracy'], best['f1'], best['auc'], best['conf_matrix'], best['epoch'] = avg_accuracy, avg_f1, auc, avg_conf_matrix, epoch
            best['predictions'], best['ground_truth'] = predictions, ground_truth
            checkpoint = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args}
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))


        # Log best epoch results
    logger = getLogger(f"root.main.{phase}.BestEpoch")
    logger.info("")
    logger.info("Printing statistics of best epoch:")
    logger.info(f"Best Epoch: {best['epoch']}")
    logger.info(f"Acc@1 {best['accuracy']:.3f}")
    logger.info(f"F1-Score {best['f1']:.3f}")
    logger.info(f"AUC ROC Score {best['f1']:.3f}")
    logger.info("")
    logger.info('Confusion Matrix:\n {}'.format(tabulate(pd.DataFrame(best['conf_matrix'],
                  columns=[f"Predicted as: {x}" for x in dataset.inverse_label_map.values()],
                  index=[f"Ground Truth: {x}" for x in dataset.inverse_label_map.values()]),
                                                         headers="keys", tablefmt='grid')))

    Logger(log_file=args.file_level_classification_log, DEBUG=args.DEBUG,
           name="root.utils.print_file_level_classification_summary",
           append_log=True if args.quantization else False, console_log=False)
    getLogger("root.utils.print_file_level_classification_summary").propagate = False
    utils.print_file_level_classification_summary(dataset_test, best['predictions'], best['ground_truth'], phase)
    logger.info(f"Generated file-level classification summary in: {args.file_level_classification_log}")

    # Export model
    logger.info('Exporting model after training.')
    if args.distributed is False or (args.distributed is True and int(os.environ['LOCAL_RANK']) == 0):
        example_input = next(iter(data_loader_test))[1]
        input_shape = (1,) + dataset.X.shape[1:]
        utils.export_model(
            model, input_shape=input_shape, output_dir=args.output_dir, opset_version=args.opset_version,
            quantization=args.quantization, example_input=example_input, generic_model=args.generic_model,
            remove_hooks_for_jit=True if (args.quantization_method == TinyMLQuantizationMethod.PTQ and args.quantization) else False)

    log_training_time(start_time)

def run(args):
    """Run training with optional distributed mode."""
    run_distributed(main_debug, args)


if __name__ == "__main__":
    arguments = get_args_parser().parse_args()
    # Apply default output_int if not specified by user
    apply_output_int_default(arguments, 'radar_classification')
    run(arguments)
