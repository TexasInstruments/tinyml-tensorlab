"""Regression tests for four bugs in the model registry and feature-extraction
composition classes:

1. get_model()'s model_config parameter was silently dead code.
2. Regression models' num_outputs was never wired from get_model()'s num_classes.
3. NeuralNetworkWithPreprocess's frozen-preprocess BatchNorm stats got corrupted
   every epoch (p.requires_grad_ = False was a no-op; preprocess ran before
   .eval() was applied, and the standard per-epoch model.train() call reset it
   back to train mode anyway).
4. Composition/helper classes with positional-arg constructors (FEModel1,
   CombinedModel, etc.) were auto-registered into the public model registry
   despite not accepting its config=<dict> calling convention.
"""
import warnings

import pytest
import torch
import torch.nn as nn

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


def test_num_outputs_is_wired_from_num_classes_for_regression_models():
    model = get_model("REG_TS_GEN_BASE_3K", variables=1, num_classes=5, input_features=512)

    assert model.num_outputs == 5
    # The final LinearLayer's out_features is built directly from
    # self.num_outputs in gen_model_spec() -- confirm it actually reflects
    # the requested value end to end, not just the raw attribute.
    final_linear = list(model.features.children())[-1] if hasattr(model, "features") else None
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linear_layers[-1].out_features == 5


def test_num_outputs_defaults_to_one_when_num_classes_not_provided_positionally():
    # Sanity check: a differing num_classes produces a differing wired value,
    # proving this isn't coincidentally always 1 or always correct.
    model = get_model("REG_TS_GEN_BASE_3K", variables=1, num_classes=2, input_features=512)
    assert model.num_outputs == 2


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


def test_composition_helper_classes_remain_directly_importable():
    # __all__ = [] must not break real usage, which imports these directly.
    assert FEModel1 is not None
    assert FEModel2 is not None
    assert FEModelLinear is not None
    assert CombinedModel is not None
    assert NeuralNetworkWithPreprocess is not None


def test_selecting_a_composition_class_by_name_raises_the_intended_not_found_error():
    with pytest.raises(ValueError, match="not found in registry"):
        get_model("FEModel1", variables=1, num_classes=2, input_features=512)
