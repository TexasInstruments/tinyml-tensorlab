"""Regression test: audio_classification's train.py and test_onnx.py must NOT
define --sample-rate. It was dead code -- GoogleSpeechCommandsDataset
(audio_dataset.py) only ever reads self.sampling_rate, set from the shared
base parser's --sampling-rate (train_base.py:113), which tinyml-modelmaker's
production orchestration (audio_base.py) already passes correctly. Keeping
--sample-rate around silently does nothing while looking like the flag that
controls audio sample rate -- a maintainability trap for anyone reading or
invoking these scripts directly."""
from tinyml_tinyverse.references.audio_classification import train, test_onnx


def _dests(parser):
    return {action.dest for action in parser._actions}


def test_train_parser_has_no_dead_sample_rate_flag():
    dests = _dests(train.get_args_parser())
    assert "sample_rate" not in dests, (
        "train.py still defines --sample-rate, but GoogleSpeechCommandsDataset "
        "never reads self.sample_rate -- only self.sampling_rate (from the "
        "shared --sampling-rate flag). This dead flag should be removed."
    )
    assert "sampling_rate" in dests, (
        "the real, working flag (--sampling-rate, from the shared base parser) "
        "must still be present."
    )


def test_test_onnx_parser_has_no_dead_sample_rate_flag():
    dests = _dests(test_onnx.get_args_parser())
    assert "sample_rate" not in dests, (
        "test_onnx.py still defines --sample-rate, but GoogleSpeechCommandsDataset "
        "never reads self.sample_rate -- only self.sampling_rate (from the "
        "shared --sampling-rate flag). This dead flag should be removed."
    )
    assert "sampling_rate" in dests, (
        "the real, working flag (--sampling-rate, from the shared base parser) "
        "must still be present."
    )
