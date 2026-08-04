"""Regression tests for three argv-builder bugs in vision/audio ai_modules:

1. image_base.py's train argv rendered data_proc_transforms as a stringified
   list while test argv passed the raw list -- prepare_transforms() (in
   tinyml-tinyverse) only combines it into args.transforms when
   isinstance(args.data_proc_transforms, list) is true, so the stringified
   form silently skipped it during training while testing still applied it.

2. audio_base.py's train and test argv builders each declared
   --data-proc-transforms/--feat-ext-transform twice (once raw, once
   stringified); argparse's last-occurrence-wins semantics meant the earlier,
   correct raw declaration was always shadowed by the later, stringified one.

3. Fixing (1) by making data_proc_transforms a raw list exposed a second bug
   an independent peer review caught: prepare_transforms() does
   `args.data_proc_transforms + args.feat_ext_transform` whenever
   data_proc_transforms is a list. image_base.py's train argv builder still
   stringified --feat-ext-transform (and --augmentation-transform), so this
   became `list + str`, raising TypeError on every image-classification
   training run. test_image_train_argv_feat_ext_transform_survives_
   prepare_transforms below exercises the REAL prepare_transforms() against
   the argv builder's actual output -- the shape-only MagicMock tests below
   couldn't catch this since they never fed argv through it.

Most of these are pure unit tests against the argv-building methods in
isolation (constructed via unittest.mock.MagicMock for self, rather than a
full ModelRunner instance) -- these methods only read attributes off
self.params and don't need real training infrastructure.
"""
from argparse import Namespace
from unittest.mock import MagicMock

from tinyml_modelmaker.ai_modules.vision.training.tinyml_tinyverse.image_base import (
    BaseImageModelTraining,
)
from tinyml_modelmaker.ai_modules.audio.training.tinyml_tinyverse.audio_base import (
    BaseAudioModelTraining,
)
from tinyml_tinyverse.references.common.train_base import prepare_transforms


def _argv_value_after(argv, flag):
    return argv[argv.index(flag) + 1]


def _count_occurrences(argv, flag):
    return argv.count(flag)


def test_image_train_argv_passes_data_proc_transforms_as_a_raw_list():
    fake_self = MagicMock()
    fake_self.params.data_processing_feature_extraction.data_proc_transforms = ["BINARIZE"]

    argv = BaseImageModelTraining._build_common_train_argv(fake_self, device="cpu", distributed=0)

    value = _argv_value_after(argv, "--data-proc-transforms")
    assert value == ["BINARIZE"]
    assert isinstance(value, list)


def test_image_train_argv_feat_ext_transform_survives_prepare_transforms():
    """Drives the REAL prepare_transforms() (tinyml-tinyverse) against the
    actual argv the train-argv builder produces -- data_proc_transforms and
    feat_ext_transform must both come out as lists, or the `+` inside
    prepare_transforms raises TypeError."""
    fake_self = MagicMock()
    fake_self.params.data_processing_feature_extraction.data_proc_transforms = ["BINARIZE"]
    fake_self.params.data_processing_feature_extraction.feat_ext_transform = ["MFCC"]

    argv = BaseImageModelTraining._build_common_train_argv(fake_self, device="cpu", distributed=0)

    args = Namespace(
        data_proc_transforms=_argv_value_after(argv, "--data-proc-transforms"),
        feat_ext_transform=_argv_value_after(argv, "--feat-ext-transform"),
    )

    prepare_transforms(args)

    assert args.transforms == ["BINARIZE", "MFCC"]


def test_image_train_and_test_argv_agree_on_data_proc_transforms_form():
    fake_self = MagicMock()
    fake_self.params.data_processing_feature_extraction.data_proc_transforms = ["BINARIZE", "RESIZE"]

    train_argv = BaseImageModelTraining._build_common_train_argv(fake_self, device="cpu", distributed=0)
    test_argv = BaseImageModelTraining._build_common_test_argv(
        fake_self, device="cpu", data_path="/tmp/data", model_path="/tmp/model.onnx", output_dir="/tmp/out"
    )

    assert _argv_value_after(train_argv, "--data-proc-transforms") == \
        _argv_value_after(test_argv, "--data-proc-transforms")


def test_audio_train_argv_declares_data_proc_transforms_exactly_once():
    fake_self = MagicMock()
    fake_self.params.data_processing_feature_extraction.data_proc_transforms = ["NORMALIZE"]
    fake_self.params.data_processing_feature_extraction.feat_ext_transform = ["MFCC"]

    argv = BaseAudioModelTraining._build_common_train_argv(fake_self, device="cpu", distributed=0)

    assert _count_occurrences(argv, "--data-proc-transforms") == 1
    assert _count_occurrences(argv, "--feat-ext-transform") == 1
    assert _argv_value_after(argv, "--data-proc-transforms") == ["NORMALIZE"]
    assert _argv_value_after(argv, "--feat-ext-transform") == ["MFCC"]


def test_audio_test_argv_declares_data_proc_transforms_exactly_once():
    fake_self = MagicMock()
    fake_self.params.data_processing_feature_extraction.data_proc_transforms = ["NORMALIZE"]
    fake_self.params.data_processing_feature_extraction.feat_ext_transform = ["MFCC"]

    argv = BaseAudioModelTraining._build_common_test_argv(
        fake_self, device="cpu", data_path="/tmp/data", model_path="/tmp/model.onnx", output_dir="/tmp/out"
    )

    assert _count_occurrences(argv, "--data-proc-transforms") == 1
    assert _count_occurrences(argv, "--feat-ext-transform") == 1
    assert _argv_value_after(argv, "--data-proc-transforms") == ["NORMALIZE"]
    assert _argv_value_after(argv, "--feat-ext-transform") == ["MFCC"]
