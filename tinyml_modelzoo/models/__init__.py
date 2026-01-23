#################################################################################
# Copyright (c) 2023-2024, Texas Instruments
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
Model registry for TinyML Model Zoo.

This module provides a unified interface to access all models in the model zoo.
Models are automatically discovered from submodules - no manual registration needed.

When adding a new model:
1. Add your model class to the appropriate file (e.g., classification.py)
2. Add the class name to that file's __all__ list
3. That's it! The model will be automatically available.
"""

import os
import importlib
import importlib.util
import inspect

# Import base class
from .base import GenericModelWithSpec

# List of model submodules to auto-discover
# Each module should export model classes via __all__
_MODEL_MODULES = [
    'classification',
    'regression',
    'anomalydetection',
    'forecasting',
    'feature_extraction',
    'image',
]

# Central model registry - built dynamically
model_dict = {}

# Track all exported names
_all_exports = ['model_dict', 'get_model', 'GenericModelWithSpec']


def _is_model_class(obj):
    """Check if an object is a model class (not an instance or function)."""
    return (
        inspect.isclass(obj) and
        (issubclass(obj, GenericModelWithSpec) or
         (hasattr(obj, 'forward') and issubclass(obj, object)))
    )


def _register_models_from_module(module_name):
    """Import a module and register all its exported models."""
    global model_dict, _all_exports

    try:
        # Import the module
        module = importlib.import_module(f'.{module_name}', package=__name__)

        # Get the __all__ list from the module
        if hasattr(module, '__all__'):
            exported_names = module.__all__
        else:
            # Fallback: export all public names that look like model classes
            exported_names = [name for name in dir(module)
                            if not name.startswith('_') and
                            _is_model_class(getattr(module, name, None))]

        # Register each exported model
        for name in exported_names:
            obj = getattr(module, name, None)
            if obj is not None:
                # Add to global namespace
                globals()[name] = obj
                _all_exports.append(name)

                # Add model classes to model_dict
                if _is_model_class(obj):
                    model_dict[name] = obj

    except ImportError as e:
        # Log but don't fail - allows graceful degradation
        import warnings
        warnings.warn(f"Could not import model module '{module_name}': {e}")


# Auto-discover and register all models from submodules
for _module_name in _MODEL_MODULES:
    _register_models_from_module(_module_name)


def import_file_or_folder(file_or_folder, package_name=None):
    """Import a Python file or folder as a module."""
    if file_or_folder is None:
        return None
    if os.path.isfile(file_or_folder):
        if package_name is None:
            package_name = os.path.splitext(os.path.basename(file_or_folder))[0]
        spec = importlib.util.spec_from_file_location(package_name, file_or_folder)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif os.path.isdir(file_or_folder):
        file_or_folder_parent = os.path.dirname(file_or_folder)
        if package_name is None:
            package_name = os.path.basename(file_or_folder)
        spec = importlib.util.spec_from_file_location(package_name,
                                                      os.path.join(file_or_folder, '__init__.py'),
                                                      submodule_search_locations=[file_or_folder_parent])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = None
    return module


def get_model(model_name, variables, num_classes, input_features=None,
              model_config=None, model_spec=None, dual_op=False):
    """
    Factory function to get a model by name.

    Args:
        model_name: Name of the model to instantiate
        variables: Number of input variables/channels
        num_classes: Number of output classes
        input_features: Input feature dimension
        model_config: Optional path to model config file
        model_spec: Optional path to custom model specification
        dual_op: Dual operation flag

    Returns:
        Instantiated model
    """
    config_dict = dict(
        variables=variables,
        num_classes=num_classes,
        input_features=input_features,
        dual_op=dual_op
    )

    # Check built-in registry first
    if model_name in model_dict:
        return model_dict[model_name](config=config_dict)

    # Try proprietary models fallback
    try:
        import tinyml_proprietary_models
        model = tinyml_proprietary_models.get_model(model_name)
        if model is not None:
            return model(config=config_dict)
    except ImportError:
        pass

    # Try loading from custom model_spec file
    if model_spec and os.path.exists(model_spec):
        model_definition = import_file_or_folder(model_spec)
        if model_definition is not None and hasattr(model_definition, 'get_model'):
            model_class = model_definition.get_model(model_name)
            if model_class is not None:
                return model_class(config=config_dict)

    raise ValueError(f"Model '{model_name}' not found in registry or specified model_spec. "
                     f"Available models: {list(model_dict.keys())}")


def list_models():
    """List all available model names."""
    return sorted(model_dict.keys())


def get_model_count():
    """Get the number of registered models."""
    return len(model_dict)


# Add utility functions to exports
_all_exports.extend(['list_models', 'get_model_count'])

# Export all public APIs dynamically
__all__ = _all_exports
