"""Regression tests for three bugs in the model registry and feature-extraction
composition classes, plus a fix-of-a-fix for a fourth:

1. get_model()'s model_config parameter was silently dead code.
2. NeuralNetworkWithPreprocess's frozen-preprocess BatchNorm stats got corrupted
   every epoch (p.requires_grad_ = False was a no-op; preprocess ran before
   .eval() was applied, and the standard per-epoch model.train() call reset it
   back to train mode anyway).
3. Composition/helper classes with positional-arg constructors (FEModel1,
   CombinedModel, etc.) were auto-registered into the public model registry
   despite not accepting its config=<dict> calling convention.
4. An earlier version of this PR "fixed" (2) by aliasing get_model()'s
   num_outputs kwarg to num_classes, and (3) by emptying
   feature_extraction.py's __all__. An independent peer review caught that
   both were wrong: num_classes is dataset.classes (folder-derived,
   unrelated to regression output count -- see
   test_num_outputs_is_not_derived_from_num_classes below), and an empty
   __all__ breaks `from tinyml_modelzoo.models import NeuralNetworkWithPreprocess`
   / `models.FEModelLinear`, both real call sites in
   {timeseries,audio,image}_classification/train.py, because the registry's
   own __all__-driven auto-discovery loop populates the package namespace
   too. See test_composition_helper_classes_are_importable_from_the_package
   and _REGISTRY_EXCLUDE in feature_extraction.py for the actual fix.
"""
import warnings

import pytest
import torch
import torch.nn as nn

import tinyml_modelzoo.models as models_pkg
from tinyml_modelzoo.models import get_model, model_dict
from tinyml_modelzoo.models.feature_extraction import (
    CombinedModel,
    FEModel1,
    FEModel2,
    FEModelLinear,
    NeuralNetworkWithPreprocess,
)


def test_model_config_warns_when_provided_and_ignored():
    with pytest.warns(UserWarning, match="model_config"):
        get_model("REG_TS_GEN_BASE_3K", variables=1, num_classes=3, input_features=512,
                   model_config="/some/config.yaml")


def test_model_config_none_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_model("REG_TS_GEN_BASE_3K", variables=1, num_classes=3, input_features=512)


def test_num_outputs_is_not_derived_from_num_classes():
    """num_classes here is dataset.classes -- for regression datasets that's
    the count of data subdirectories (BaseGenericTSDataset._get_classes(),
    which GenericTSDatasetReg never overrides), not the number of regression
    targets. Aliasing num_outputs=num_classes (an earlier version of this
    fix) silently reshapes the model's output head to match subdirectory
    count instead of the model's own correct hardcoded default, breaking any
    regression dataset with more than one class subdirectory. num_outputs
    must stay at the model's own default (1 for REG_TS_GEN_BASE_3K)
    regardless of what num_classes is."""
    model_low = get_model("REG_TS_GEN_BASE_3K", variables=1, num_classes=2, input_features=512)
    model_high = get_model("REG_TS_GEN_BASE_3K", variables=1, num_classes=5, input_features=512)

    assert model_low.num_outputs == 1
    assert model_high.num_outputs == 1


class _TinyPreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.conv = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x):
        return self.bn(self.conv(x))


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x.mean(dim=(2, 3)))


def test_neural_network_with_preprocess_freezes_gradients():
    preprocess = _TinyPreprocess()
    model = _TinyModel()
    wrapper = NeuralNetworkWithPreprocess(preprocess, model)

    assert all(not p.requires_grad for p in wrapper.preprocess.parameters())
    # The trainable model's own params must be unaffected.
    assert all(p.requires_grad for p in wrapper.model.parameters())


def test_neural_network_with_preprocess_stays_in_eval_after_outer_train_call():
    """The real-world failure mode: the training loop calls model.train() once
    per epoch. Without the fix, this recursively resets preprocess back to
    train mode too, so its BatchNorm running stats get updated by the next
    forward pass -- silently corrupting the "frozen" feature extractor."""
    preprocess = _TinyPreprocess()
    model = _TinyModel()
    wrapper = NeuralNetworkWithPreprocess(preprocess, model)

    wrapper.train()  # what the training loop calls every epoch

    assert wrapper.preprocess.training is False
    assert wrapper.preprocess.bn.training is False
    # The trainable model must still actually be in train mode.
    assert wrapper.model.training is True


def test_neural_network_with_preprocess_bn_running_stats_unchanged_by_forward():
    preprocess = _TinyPreprocess()
    model = _TinyModel()
    wrapper = NeuralNetworkWithPreprocess(preprocess, model)
    wrapper.train()

    running_mean_before = wrapper.preprocess.bn.running_mean.clone()
    running_var_before = wrapper.preprocess.bn.running_var.clone()

    x = torch.randn(4, 3, 8, 8)
    wrapper(x)

    assert torch.equal(wrapper.preprocess.bn.running_mean, running_mean_before)
    assert torch.equal(wrapper.preprocess.bn.running_var, running_var_before)


def test_neural_network_with_preprocess_without_model_is_not_frozen():
    """Matches the real call pattern NeuralNetworkWithPreprocess(fe_model, None)
    -- preprocess used standalone should remain trainable, preserving the
    original conditional (only freeze when both preprocess and model exist)."""
    preprocess = _TinyPreprocess()
    wrapper = NeuralNetworkWithPreprocess(preprocess, None)

    assert all(p.requires_grad for p in wrapper.preprocess.parameters())
    wrapper.train()
    assert wrapper.preprocess.training is True


def test_composition_helper_classes_are_not_in_the_public_registry():
    for name in ("FEModel1", "FEModel2", "FEModel", "FEModelLinear", "CombinedModel",
                 "NeuralNetworkWithPreprocess"):
        assert name not in model_dict, (
            f"{name} should not be selectable via get_model()/model_dict -- "
            "its constructor doesn't accept the config=<dict> convention."
        )


def test_composition_helper_classes_are_importable_from_the_package():
    """These names must survive the registry's __all__-driven auto-discovery
    loop at the package level, not just as direct submodule imports -- real
    callers use `from tinyml_tinyverse.common.models import
    NeuralNetworkWithPreprocess` (which re-exports via
    `from tinyml_modelzoo.models import *`) and `models.FEModelLinear(...)`
    attribute access (audio/image/timeseries_classification/train.py)."""
    assert models_pkg.NeuralNetworkWithPreprocess is NeuralNetworkWithPreprocess
    assert models_pkg.FEModelLinear is FEModelLinear
    assert models_pkg.FEModel1 is FEModel1
    assert models_pkg.FEModel2 is FEModel2
    assert models_pkg.CombinedModel is CombinedModel
    assert "NeuralNetworkWithPreprocess" in models_pkg.__all__
    assert "FEModelLinear" in models_pkg.__all__


def test_selecting_a_composition_class_by_name_raises_the_intended_not_found_error():
    with pytest.raises(ValueError, match="not found in registry"):
        get_model("FEModel1", variables=1, num_classes=2, input_features=512)
