"""Checks scripts/fe_rules.py against the real tinyverse dataset class.

Run: python tests/test_fe_rules.py   (or pytest). The equivalence tests are skipped when
tinyml_tinyverse is not importable; the rule tests always run.
"""
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plugins", "tinyml-agent-skills",
                                "skills", "ti-ccstudio-tinyml-workflow", "scripts"))
import fe_rules as fr  # noqa: E402

FFTBIN = ["FFT_FE", "FFT_POS_HALF", "DC_REMOVE", "ABS", "BINNING", "LOG_DB", "CONCAT"]
FFT = ["FFT_FE", "FFT_POS_HALF", "ABS", "DC_REMOVE", "LOG_DB", "CONCAT"]
RAW = ["RAW_FE", "CONCAT"]
Q15 = ["FFT_Q15", "Q15_SCALE", "Q15_MAG", "DC_REMOVE", "BIN_Q15", "CONCAT"]


def _real_shape(R, V, F, B, C, S, A, m, stacking, transforms):
    """X.shape[1:] and len(X) from the real GenericTSDataset feature extraction."""
    from tinyml_tinyverse.common.datasets.timeseries_dataset import GenericTSDataset
    d = object.__new__(GenericTSDataset)
    d.logger = logging.getLogger("t")
    d.transforms = d.feat_ext_transform = transforms
    d.variables, d.frame_size, d.feature_size_per_frame = V, F, B
    d.num_frame_concat, d.frame_skip, d.stacking, d.min_bin, d.analysis_bandwidth = C, S, stacking, m, A
    d.normalize_bin, d.offset, d.scale = True, 0, 1
    d.log_mul, d.log_base, d.log_threshold, d.dc_remove, d.sampling_rate = 20, 10, 1e-100, True, 1
    d.q15_scale_factor, d.window_count, d.chunk_size, d.fft_size, d.stride_size = 4, 1, 1, 1, 0.01
    d.classes, d.feature_extraction_params, d.preprocessing_flags = ["a"], {}, []
    d._prepare_feature_extraction_variables()
    X, _ = d._feature_extraction(np.random.randn(R, V).T, "f.csv")
    return len(X), tuple(X.shape[1:])


CASES = [
    # R, V, F, B, C, S, A, m, stacking, transforms
    (1000, 1, 32, 4, 8, 1, 1, 1, "2D1", FFTBIN),
    (1000, 3, 32, 4, 8, 1, 1, 1, "2D1", FFTBIN),
    (1000, 3, 32, 4, 8, 1, 1, 1, "1D", FFTBIN),
    (1000, 3, 32, 4, 8, 2, 1, 1, "2D1", FFTBIN),
    (1000, 3, 64, 8, 4, 1, 1, 1, "2D1", FFTBIN),
    (1000, 1, 64, 8, 4, 3, 1, None, "2D1", FFTBIN),
    (1000, 1, 64, None, 1, 1, 1, 1, "2D1", FFT),
    (1000, 1, 64, None, 4, 1, 1, 0, "2D1", FFT),
    (1000, 2, 128, 16, 4, 2, 1, 1, "2D1", FFTBIN),
    (1000, 3, 128, 128, 1, 1, 1, 1, "2D1", RAW),
    (1000, 1, 64, 4, 3, 1, 2, 1, "2D1", FFTBIN),
]


def test_fit_matches_dataset():
    try:
        import tinyml_tinyverse  # noqa: F401
    except ImportError:
        print("skip: tinyml_tinyverse not installed")
        return
    for R, V, F, B, C, S, A, m, stk, tr in CASES:
        n, shape = _real_shape(R, V, F, B, C, S, A, m, stk, tr)
        got = fr.fit(R, F, C, S, V, stk, tr, B, A, m)
        assert (n, shape) == (got["windows"], got["input_shape"]), (R, V, F, B, C, S, A, m, stk, got, n, shape)


def test_simple_window_path():
    try:
        from tinyml_tinyverse.common.transforms import basic_transforms as bt
    except ImportError:
        print("skip: tinyml_tinyverse not installed")
        return
    for R, V, F, C, stride, B in [(1000, 1, 32, 8, 8, 4), (1000, 3, 32, 4, 16, 4), (300, 1, 64, 2, 64, 8),
                                  (2000, 2, 64, 4, 10, 8), (120, 1, 32, 8, 8, 4), (95, 1, 32, 8, 8, 4)]:
        rows = np.array(bt.SimpleWindow(np.random.randn(R, V), F, stride, False)).reshape(-1, V).shape[0]
        n, shape = _real_shape(rows, V, F, B, C, 1, 1, 1, "2D1", FFTBIN)
        got = fr.fit(R, F, C, 1, V, "2D1", FFTBIN, B, 1, 1, window_stride=stride)
        assert (n, shape) == (got["windows"], got["input_shape"]), (R, V, F, C, stride, got, n, shape)


def test_q15_chain_length():
    try:
        import cmsisdsp  # noqa: F401
        import tinyml_tinyverse  # noqa: F401
    except ImportError:
        print("skip: cmsisdsp / tinyml_tinyverse not installed")
        return
    n, shape = _real_shape(2560, 1, 256, 16, 8, 1, 1, 1, "2D1", Q15)
    got = fr.fit(2560, 256, 8, 1, 1, "2D1", Q15, 16, 1, 1)
    assert (n, shape) == (got["windows"], got["input_shape"])


def test_min_rows_required():
    assert fr.min_rows_required(32, 8) == 256
    assert fr.fit(256, 32, 8, transforms=FFTBIN, feature_size_per_frame=4)["windows"] == 1
    assert "skipped" in fr.fit(255, 32, 8, transforms=FFTBIN, feature_size_per_frame=4)
    # a file longer than one frame but shorter than frame*concat passes a frame-only check and is skipped
    assert "skipped" in fr.fit(100, 32, 8, transforms=FFTBIN, feature_size_per_frame=4)
    # Downsample by 2 doubles the rows needed; SimpleWindow with stride 8: F + (C-1)*stride
    assert fr.min_rows_required(32, 8, downsample=2) == 2 * 255 + 1
    assert fr.min_rows_required(32, 8, window_stride=8) == 32 + 7 * 8
    r = fr.min_rows_required(32, 8, window_stride=8)
    assert fr.fit(r, 32, 8, transforms=FFTBIN, feature_size_per_frame=4, window_stride=8)["windows"] == 1
    assert "skipped" in fr.fit(r - 1, 32, 8, transforms=FFTBIN, feature_size_per_frame=4, window_stride=8)
    assert fr.min_rows_required(32, 1, has_fe=False) is None


def test_pipeline_rules():
    e, w = fr.pipeline_issues("generic_timeseries_classification", FFTBIN, frame_size=64, feature_size_per_frame=8,
                              num_frame_concat=4, min_bin=1, using_preset=True)
    assert not e, e
    e, _ = fr.pipeline_issues("generic_timeseries_classification", [t for t in FFTBIN if t != "CONCAT"],
                              frame_size=64, feature_size_per_frame=8, num_frame_concat=4, using_preset=True)
    assert any("CONCAT" in x for x in e)
    e, w = fr.pipeline_issues("generic_timeseries_classification", FFT, frame_size=64, num_frame_concat=1)
    assert not e and any("log_threshold" in x for x in w), (e, w)   # unset is a warning, not an error
    e, _ = fr.pipeline_issues("generic_timeseries_classification", FFTBIN, frame_size=64, feature_size_per_frame=40,
                              log_threshold=1e-100)
    assert any("bin_size" in x for x in e)
    e, _ = fr.pipeline_issues("generic_timeseries_forecasting", FFT)
    assert any("Forecasting" in x for x in e)
    e, _ = fr.pipeline_issues("generic_timeseries_classification", FFT, frame_size=100, log_threshold=1e-100)
    assert any("power of two" in x for x in e)
    e, _ = fr.pipeline_issues("generic_timeseries_classification", ["FE_DECIMATE", "FE_Q15", "CONCAT"])
    assert any("no effect" in x for x in e)
    _, w = fr.pipeline_issues("generic_timeseries_classification", FFTBIN, frame_size=32, feature_size_per_frame=4,
                              min_bin=1, log_threshold=1e-100)
    assert any("last band" in x for x in w), w
    _, w = fr.pipeline_issues("generic_timeseries_classification", FFTBIN, frame_size=32, feature_size_per_frame=4,
                              min_bin=None, log_threshold=1e-100)
    assert not any("last band" in x for x in w), w


def test_device_and_shapes():
    assert fr.device_family("MSPM0G5187") == "mspm0" and fr.device_family("F28P55") == "other"
    assert fr.device_family(None) is None
    assert fr.output_shape(FFTBIN, 32, 8, 3, "2D1", 4)["shape"] == (3, 32, 1)
    assert fr.output_shape(FFTBIN, 32, 8, 3, "1D", 4)["shape"] == (1, 96, 1)
    assert fr.output_shape(["PIR_FE_Q15"], 125, 1, 1)["shape"] == (1, 25, 25)
    assert fr.output_shape(["PIR_FE"], 125, 1, 1, window_count=25, chunk_size=8, stride_size=0.032)["shape"] == (1, 25, 25)
    assert fr.output_shape([], 256, 1, 3)["shape"] == (3, 256, 1)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
