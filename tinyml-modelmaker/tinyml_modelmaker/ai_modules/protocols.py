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

"""Protocol definitions for tinyml-modelmaker component interfaces.

These protocols document the implicit contracts that ModelRunner, ModelTraining,
ModelCompilation, and DatasetHandling implementations must satisfy.  They use
structural subtyping (typing.Protocol) so existing classes conform automatically
without inheriting from them.

Usage with static type checkers (mypy / pyright)::

    from tinyml_modelmaker.ai_modules.protocols import Trainer

    def start_training(trainer: Trainer) -> None:
        trainer.clear()
        trainer.run()

Runtime checks are also supported via ``@runtime_checkable``::

    isinstance(my_training_obj, Trainer)  # True if it has the right methods
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..utils.config_dict import ConfigDict


# ---------------------------------------------------------------------------
# Base protocol shared by all pipeline components
# ---------------------------------------------------------------------------

@runtime_checkable
class LifecycleComponent(Protocol):
    """Base protocol for pipeline components.

    Every component in the tinyml-modelmaker pipeline follows the same
    lifecycle: ``init_params()`` -> ``__init__()`` -> ``clear()`` ->
    ``run()`` -> ``get_params()``.  This protocol captures the subset
    of that lifecycle that is common to *all* component types.

    Note: All concrete implementations also store a ``params: ConfigDict``
    instance attribute.  It is omitted here so that ``@runtime_checkable``
    ``issubclass()`` checks work (Python disallows non-method members in
    runtime-checkable protocol ``issubclass()`` calls).  Static type
    checkers enforce the attribute via the child protocols' ``__init__``
    signatures.
    """

    @classmethod
    def init_params(cls, *args: Any, **kwargs: Any) -> ConfigDict: ...

    def clear(self) -> None: ...

    def get_params(self) -> ConfigDict: ...


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

@runtime_checkable
class DatasetHandler(LifecycleComponent, Protocol):
    """Protocol for dataset handling components.

    Concrete implementation: ``common.datasets.DatasetHandling``
    """

    def __init__(self, *args: Any, quit_event: Any = None, **kwargs: Any) -> None: ...

    def run(self) -> None: ...


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

@runtime_checkable
class Trainer(LifecycleComponent, Protocol):
    """Protocol for model training components.

    Concrete implementations:
        - ``timeseries.training.tinyml_tinyverse.timeseries_classification.ModelTraining``
        - ``timeseries.training.tinyml_tinyverse.timeseries_regression.ModelTraining``
        - ``timeseries.training.tinyml_tinyverse.timeseries_anomalydetection.ModelTraining``
        - ``timeseries.training.tinyml_tinyverse.timeseries_forecasting.ModelTraining``
        - ``vision.training.tinyml_tinyverse.image_classification.ModelTraining``
    """

    def __init__(self, *args: Any, quit_event: Any = None, **kwargs: Any) -> None: ...

    def run(self, **kwargs: Any) -> None: ...

    def stop(self) -> None: ...


# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------

@runtime_checkable
class Compiler(LifecycleComponent, Protocol):
    """Protocol for model compilation components.

    Concrete implementation: ``common.compilation.tinyml_benchmark.ModelCompilation``
    """

    def __init__(self, *args: Any, quit_event: Any = None, **kwargs: Any) -> None: ...

    def run(self, **kwargs: Any) -> int: ...


# ---------------------------------------------------------------------------
# Top-level model runner
# ---------------------------------------------------------------------------

@runtime_checkable
class Runner(LifecycleComponent, Protocol):
    """Protocol for the top-level model runner.

    Concrete implementations:
        - ``timeseries.runner.ModelRunner``
        - ``vision.runner.ModelRunner``
    """

    def __init__(self, *args: Any, verbose: bool = True, **kwargs: Any) -> None: ...

    def prepare(self) -> str: ...

    def run(self) -> ConfigDict: ...

    def write_status_file(self) -> str: ...

    def package_trained_model(self, input_files: list, compressed_file_name: str) -> int: ...
