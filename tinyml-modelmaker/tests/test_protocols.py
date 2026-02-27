"""Tests that concrete classes satisfy the Protocol definitions in ai_modules.protocols.

Uses ``@runtime_checkable`` isinstance checks to verify that every component
implementation provides the methods required by its corresponding protocol.

Requires TVM (compilation backend) which is not available on all platforms.
"""

import pytest

# TVM is required transitively via tinyml_benchmark → compilation.py
tvm = pytest.importorskip("tvm", reason="TVM not available on this platform")

from tinyml_modelmaker.ai_modules.protocols import (
    Compiler,
    DatasetHandler,
    LifecycleComponent,
    Runner,
    Trainer,
)


# ---------------------------------------------------------------------------
# Concrete class imports (all mocked via conftest.py)
# ---------------------------------------------------------------------------

from tinyml_modelmaker.ai_modules.timeseries.runner import (
    ModelRunner as TimeseriesModelRunner,
)
from tinyml_modelmaker.ai_modules.vision.runner import (
    ModelRunner as VisionModelRunner,
)
from tinyml_modelmaker.ai_modules.timeseries.training.tinyml_tinyverse.timeseries_classification import (
    ModelTraining as TSClassificationTraining,
)
from tinyml_modelmaker.ai_modules.timeseries.training.tinyml_tinyverse.timeseries_regression import (
    ModelTraining as TSRegressionTraining,
)
from tinyml_modelmaker.ai_modules.timeseries.training.tinyml_tinyverse.timeseries_anomalydetection import (
    ModelTraining as TSAnomalyDetectionTraining,
)
from tinyml_modelmaker.ai_modules.timeseries.training.tinyml_tinyverse.timeseries_forecasting import (
    ModelTraining as TSForecastingTraining,
)
from tinyml_modelmaker.ai_modules.vision.training.tinyml_tinyverse.image_classification import (
    ModelTraining as VisionClassificationTraining,
)
from tinyml_modelmaker.ai_modules.common.compilation.tinyml_benchmark import (
    ModelCompilation,
)
from tinyml_modelmaker.ai_modules.common.datasets import DatasetHandling


# ===================================================================
# Protocol conformance tests
# ===================================================================


class TestRunnerProtocol:
    """Verify that ModelRunner classes satisfy the Runner protocol."""

    def test_timeseries_runner_satisfies_runner_protocol(self):
        assert issubclass(TimeseriesModelRunner, Runner)

    def test_vision_runner_satisfies_runner_protocol(self):
        assert issubclass(VisionModelRunner, Runner)

    def test_timeseries_runner_satisfies_lifecycle_protocol(self):
        assert issubclass(TimeseriesModelRunner, LifecycleComponent)

    def test_vision_runner_satisfies_lifecycle_protocol(self):
        assert issubclass(VisionModelRunner, LifecycleComponent)


class TestTrainerProtocol:
    """Verify that ModelTraining classes satisfy the Trainer protocol."""

    def test_timeseries_classification_satisfies_trainer_protocol(self):
        assert issubclass(TSClassificationTraining, Trainer)

    def test_timeseries_regression_satisfies_trainer_protocol(self):
        assert issubclass(TSRegressionTraining, Trainer)

    def test_timeseries_anomalydetection_satisfies_trainer_protocol(self):
        assert issubclass(TSAnomalyDetectionTraining, Trainer)

    def test_timeseries_forecasting_satisfies_trainer_protocol(self):
        assert issubclass(TSForecastingTraining, Trainer)

    def test_vision_classification_satisfies_trainer_protocol(self):
        assert issubclass(VisionClassificationTraining, Trainer)


class TestCompilerProtocol:
    """Verify that ModelCompilation satisfies the Compiler protocol."""

    def test_compilation_satisfies_compiler_protocol(self):
        assert issubclass(ModelCompilation, Compiler)

    def test_compilation_satisfies_lifecycle_protocol(self):
        assert issubclass(ModelCompilation, LifecycleComponent)


class TestDatasetHandlerProtocol:
    """Verify that DatasetHandling satisfies the DatasetHandler protocol."""

    def test_dataset_handling_satisfies_dataset_handler_protocol(self):
        assert issubclass(DatasetHandling, DatasetHandler)

    def test_dataset_handling_satisfies_lifecycle_protocol(self):
        assert issubclass(DatasetHandling, LifecycleComponent)


class TestProtocolRuntimeCheckable:
    """Verify that protocols work correctly for non-conforming classes."""

    def test_lifecycle_component_is_runtime_checkable(self):
        """LifecycleComponent should be usable with isinstance/issubclass."""
        assert isinstance(LifecycleComponent, type)

    def test_non_conforming_class_fails_isinstance_check(self):
        """A class missing required methods should NOT satisfy the protocol."""

        class Incomplete:
            pass

        assert not issubclass(Incomplete, LifecycleComponent)
        assert not issubclass(Incomplete, Trainer)
        assert not issubclass(Incomplete, Compiler)
        assert not issubclass(Incomplete, DatasetHandler)
        assert not issubclass(Incomplete, Runner)

    def test_partial_conformance_fails(self):
        """A class with only some of the required methods should still fail."""

        class PartialComponent:
            def clear(self):
                pass

            def get_params(self):
                pass

        # Missing init_params classmethod — should fail LifecycleComponent
        assert not issubclass(PartialComponent, Trainer)
        assert not issubclass(PartialComponent, Runner)
