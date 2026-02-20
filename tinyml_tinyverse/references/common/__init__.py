#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
# All Rights Reserved.
#################################################################################

"""
Common module for timeseries reference scripts.
Contains shared functionality for training and testing across task types.
"""

from .train_base import (
    # Argument parsing
    split_weights,
    get_base_args_parser,
    # Golden vector generation
    generate_golden_vector_dir,
    generate_user_input_config,
    generate_test_vector,
    generate_model_aux,
    create_golden_vectors_base,
    assemble_golden_vectors_header,
    # Dataset loading
    load_datasets,
    # Training environment setup
    setup_training_environment,
    prepare_transforms,
    create_data_loaders,
    # Model creation and setup
    create_model,
    log_model_summary,
    load_pretrained_weights,
    move_model_to_device,
    # Optimizer and distributed setup
    setup_optimizer_and_scheduler,
    setup_distributed_model,
    resume_from_checkpoint,
    # Training utilities
    save_checkpoint,
    handle_export_only,
    export_trained_model,
    log_training_time,
    compute_default_output_int,
    apply_output_int_default,
    get_output_int_flag,
    load_onnx_for_inference,
    # Distributed training
    run_distributed,
)

from .test_onnx_base import (
    get_base_test_args_parser,
    setup_test_environment,
    prepare_transforms as prepare_test_transforms,
    load_onnx_model,
    run_distributed_test,
)
