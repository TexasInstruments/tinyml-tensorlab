#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
quantize_onnx.py

Quantize an ONNX model with random calibration data.  In addition to the
original functionality this version prints the number of model parameters
and the total size (MiB) occupied by those parameters.

Usage:
    python quantize_onnx.py <model.onnx> [--out-file <quantized.onnx>] [--epochs <N>]

If ``--out-file`` is omitted the script creates ``<model_name>_int8.onnx`` next
to the source. ``--epochs`` controls how many random batches are generated for
calibration (default = 10).
"""

import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    CalibrationMethod,
)
from onnx import numpy_helper

# ----------------------------------------------------------------------
# Mapping from ONNX type strings to NumPy dtypes (same as in infer_with_onnx.py)
# ----------------------------------------------------------------------
ORT_TYPE_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
    # add more if you hit an unsupported type
}


def _make_random_input(shape, np_dtype):
    """Create a random NumPy array that matches the requested ONNX shape/dtype."""
    # Replace dynamic dimensions (None / symbolic) with 1 so the array is concrete.
    concrete_shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in shape)

    if np.issubdtype(np_dtype, np.floating):
        return np.random.randn(*concrete_shape).astype(np_dtype)
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        low = max(info.min, -10)
        high = min(info.max, 10)
        return np.random.randint(low, high + 1, size=concrete_shape, dtype=np_dtype)
    if np.issubdtype(np_dtype, np.bool_):
        return np.random.choice([False, True], size=concrete_shape)

    raise RuntimeError(f"Unsupported NumPy dtype {np_dtype}")


class RandomCalibrationReader(CalibrationDataReader):
    """
    ONNX Runtime expects an iterator that yields a dict
    {input_name: np.ndarray, ...} for each calibration batch.
    This implementation creates ``epochs`` batches of random data on‑the‑fly.
    """

    def __init__(self, input_meta, epochs: int = 10):
        """
        Parameters
        ----------
        input_meta : list[ort.NodeInfo]
            Output of ``session.get_inputs()`` – contains name, shape, type.
        epochs : int
            Number of random batches to generate.
        """
        self.input_meta = input_meta
        self.epochs = epochs
        self._cnt = 0

    def get_next(self):
        if self._cnt >= self.epochs:
            return None

        feed = {}
        for meta in self.input_meta:
            name = meta.name
            shape = meta.shape
            onnx_type = meta.type
            np_dtype = ORT_TYPE_TO_NP.get(onnx_type)

            if np_dtype is None:
                raise RuntimeError(
                    f"Unsupported ONNX type '{onnx_type}' for input '{name}'"
                )

            feed[name] = _make_random_input(shape, np_dtype)

        self._cnt += 1
        return feed


def _print_model_parameter_stats(onnx_model: onnx.ModelProto):
    """
    Print total number of parameters and the memory they occupy.

    Parameters
    ----------
    onnx_model : onnx.ModelProto
        Loaded ONNX model.
    """
    total_params = 0
    total_bytes = 0

    for initializer in onnx_model.graph.initializer:
        # Convert the initializer to a NumPy array to get its size & dtype.
        np_arr = numpy_helper.to_array(initializer)
        numel = np_arr.size
        total_params += int(numel)
        total_bytes += np_arr.nbytes

    # Convert bytes → MiB (1 MiB = 2**20 bytes)
    total_mib = total_bytes / (1024 ** 2)

    print("\n=== Model Parameter Statistics ===")
    print(f"  → Total parameters : {total_params:,}")
    print(f"  → Size of parameters: {total_mib:.2f} MiB")
    print("===================================\n")


def quant_onnx(
    onnx_path: str,
    out_path: str = None,
    epochs: int = 10,
    quant_method: CalibrationMethod = CalibrationMethod.MinMax,
):
    """
    Perform static INT8 quantization on an ONNX model using random inputs.

    Parameters
    ----------
    onnx_path : str
        Path to the original (float) ONNX model.
    out_path : str | None
        Where to write the quantized model.  If ``None`` a file named
        ``<model_name>_int8.onnx`` will be placed next to the source.
    epochs : int
        Number of random calibration batches (default 10).
    quant_method : onnxruntime.quantization.CalibrationMethod
        Calibration algorithm – ``MinMax`` (default) or ``Entropy``.
    """
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # --------------------------------------------------
    # 1️⃣ Load the model (sanity check) and print stats
    # --------------------------------------------------
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    _print_model_parameter_stats(model)

    # --------------------------------------------------
    # 2️⃣ Create an ONNX Runtime session to read input meta‑data
    # --------------------------------------------------
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()

    # --------------------------------------------------
    # 3️⃣ Build the random calibration data‑reader
    # --------------------------------------------------
    calib_reader = RandomCalibrationReader(input_meta, epochs=epochs)

    # --------------------------------------------------
    # 4️⃣ Determine output path
    # --------------------------------------------------
    if out_path is None:
        base_dir, model_file = os.path.split(onnx_path)
        model_name, _ = os.path.splitext(model_file)
        out_path = os.path.join(base_dir, f"{model_name}_int8.onnx")

    # --------------------------------------------------
    # 5️⃣ Run static quantization
    # --------------------------------------------------
    print(f"\n=== Quantizing {onnx_path} ===")
    print(f"  → Calibration batches (epochs): {epochs}")
    print(f"  → Output model: {out_path}")

    quantize_static(
        model_input=onnx_path,
        model_output=out_path,
        calibration_data_reader=calib_reader,
        quant_format=ort.quantization.QuantFormat.QOperator,
        activation_type=ort.quantization.QuantType.QInt8,
        weight_type=ort.quantization.QuantType.QInt8,
        calibrate_method=quant_method,
        per_channel=True,
        reduce_range=False,
    )

    print("\n✅ Quantization completed.")
    return out_path


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------
    # Simple argument parsing (no external deps)
    # --------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python quantize_onnx.py <model.onnx> --out-file <quantized.onnx> [--epochs <N>]")
        sys.exit(1)

    model_path = sys.argv[1]
    out_file = None
    epochs = 10  # default

    if "--out-file" in sys.argv:
        idx = sys.argv.index("--out-file")
        if idx + 1 >= len(sys.argv):
            raise ValueError("Missing argument after '--out-file'")
        out_file = sys.argv[idx + 1]

    if "--epochs" in sys.argv:
        idx = sys.argv.index("--epochs")
        if idx + 1 >= len(sys.argv):
            raise ValueError("Missing argument after '--epochs'")
        try:
            epochs = int(sys.argv[idx + 1])
        except ValueError:
            raise ValueError("'--epochs' argument must be an integer")

    quant_onnx(model_path, out_path=out_file, epochs=epochs)