# tinyml-modelmaker/scripts/infer_with_onnx.py
# ------------------------------------------------------------
# Copyright (c) 2023‑2024, Texas Instruments
# ------------------------------------------------------------

import os
import sys
import csv
import numpy as np
import onnxruntime as ort
import onnx
# ------------------------------------------------------------
# TinyML ONNX inference helper – model‑info dump
# ------------------------------------------------------------
import os
import sys
import csv
import numpy as np
import onnxruntime as ort
import onnx
import json

def _tensor_num_elements(tensor):
    """Return the total number of elements of an ONNX TensorProto."""
    # The shape is stored as a list of ints; a scalar has an empty list.
    shape = list(tensor.dims)
    return int(np.prod(shape)) if shape else 1

def _tensor_byte_size(tensor):
    """Return the size in bytes of an ONNX TensorProto."""
    # Map ONNX data types to numpy dtypes (only the common ones needed here)
    onnx_to_np = {
        onnx.TensorProto.FLOAT:      np.dtype('float32'),
        onnx.TensorProto.UINT8:      np.dtype('uint8'),
        onnx.TensorProto.INT8:       np.dtype('int8'),
        onnx.TensorProto.UINT16:     np.dtype('uint16'),
        onnx.TensorProto.INT16:      np.dtype('int16'),
        onnx.TensorProto.UINT32:     np.dtype('uint32'),
        onnx.TensorProto.INT32:      np.dtype('int32'),
        onnx.TensorProto.UINT64:     np.dtype('uint64'),
        onnx.TensorProto.INT64:      np.dtype('int64'),
        onnx.TensorProto.DOUBLE:     np.dtype('float64'),
        onnx.TensorProto.BOOL:       np.dtype('bool'),
    }
    np_dtype = onnx_to_np.get(tensor.data_type, None)
    if np_dtype is None:
        raise RuntimeError(f"Unsupported ONNX dtype {tensor.data_type}")

    return _tensor_num_elements(tensor) * np_dtype.itemsize

def dump_onnx_model_info(onnx_path: str, info_path: str):
    """
    Create ``<model_name>_info.txt`` next to ``onnx_path`` containing:
        * total number of parameters
        * total memory occupied by parameters (bytes)
        * per‑layer (node) parameter memory (bytes)
    """
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # --------------------------------------------------
    # 1️⃣ Load the model
    # --------------------------------------------------
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    # --------------------------------------------------
    # 2️⃣ Build a map: initializer name → (bytes, element count)
    # --------------------------------------------------
    init_map = {}
    total_params = 0
    total_bytes  = 0
    for init in model.graph.initializer:
        n_elem = _tensor_num_elements(init)
        n_bytes = _tensor_byte_size(init)
        init_map[init.name] = (n_bytes, n_elem)
        total_params += n_elem
        total_bytes  += n_bytes

    # --------------------------------------------------
    # 3️⃣ Per‑node memory consumption
    # --------------------------------------------------
    # Some nodes (e.g. Conv, Gemm) will reference one or more
    # initializers as their weight/bias tensors.  We sum the size of
    # every initializer that appears in a node’s input list.
    node_mem = []   # list of (node_name, bytes, param_cnt)
    for node in model.graph.node:
        node_bytes = 0
        node_params = 0
        for inp in node.input:
            if inp in init_map:
                b, p = init_map[inp]
                node_bytes += b
                node_params += p
        # Use the node’s output name as a readable identifier if possible
        node_id = node.name or (node.output[0] if node.output else "UnnamedNode")
        node_mem.append((node_id, node_bytes, node_params))

    # --------------------------------------------------
    # 4️⃣ Write the info file
    # --------------------------------------------------
    base_dir, model_file = os.path.split(onnx_path)
    model_name, _ = os.path.splitext(model_file)
    if not info_path:
        info_path = os.path.join(base_dir, f"{model_name}_info.txt")

    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_file}\n")
        f.write(f"Total parameters : {total_params:,}\n")
        f.write(f"Total param memory (bytes) : {total_bytes:,}\n")
        f.write("\nLayer‑by‑layer parameter memory:\n")
        f.write(f"{'Layer':30} {'Parameters':>15} {'Memory (bytes)':>15}\n")
        f.write("-" * 65 + "\n")
        for layer, mem_bytes, mem_params in node_mem:
            f.write(f"{layer:30} {mem_params:15,} {mem_bytes:15,}\n")

    print(f"[INFO] Model statistics written to: {info_path}")

# ------------------------------------------------------------
# Helper: map ONNX type strings (e.g. "tensor(float)") to NumPy dtypes
# ------------------------------------------------------------
ORT_TYPE_TO_NP = {
    "tensor(float)":   np.float32,
    "tensor(double)":  np.float64,
    "tensor(int8)":    np.int8,
    "tensor(int16)":   np.int16,
    "tensor(int32)":   np.int32,
    "tensor(int64)":   np.int64,
    "tensor(uint8)":   np.uint8,
    "tensor(uint16)":  np.uint16,
    "tensor(uint32)":  np.uint32,
    "tensor(uint64)":  np.uint64,
    "tensor(bool)":    np.bool_,
    # add more if needed
}


def _resolve_dim(dim):
    """Replace a dynamic dimension (None, 0 or a symbolic name) with a concrete size."""
    if isinstance(dim, (int, np.integer)):
        return dim if dim > 0 else 1
    # Symbolic dimensions (e.g. "batch") – fall back to 1
    return 1


def _make_random_input(shape, np_dtype):
    """Create a random NumPy array that respects *shape* and *np_dtype*."""
    concrete_shape = tuple(_resolve_dim(d) for d in shape)

    if np.issubdtype(np_dtype, np.floating):
        return np.random.random(concrete_shape).astype(np_dtype)
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        low = max(info.min, -10)
        high = min(info.max, 10)
        return np.random.randint(low, high + 1, size=concrete_shape, dtype=np_dtype)
    if np_dtype is np.bool_:
        return np.random.choice([False, True], size=concrete_shape)
    raise RuntimeError(f"Unsupported dtype {np_dtype}")


def _save_io_csv(inputs: dict, outputs: list, csv_path: str):
    """
    Persist all input and output tensors to a single CSV file.

    The CSV layout is simple:
        * each **row** corresponds to one tensor (input or output)
        * the **first column** holds the tensor name (prefixed with ``in_`` or ``out_``)
        * the remaining columns contain the flattened values of that tensor

    This format is easy to read back with ``pandas.read_csv`` or the built‑in ``csv`` module.
    """
    with open(csv_path, mode="w", newline="") as fp:
        writer = csv.writer(fp)

        # ---- write inputs -------------------------------------------------
        for name, arr in inputs.items():
            flat = arr.ravel().tolist()
            writer.writerow([f"in_{name}"] + flat)

        # ---- write outputs ------------------------------------------------
        for idx, arr in enumerate(outputs):
            # Try to keep the original ONNX output name if available later;
            # here we just use a generic index‑based name.
            writer.writerow([f"out_{idx}"] + arr.ravel().tolist())

    print(f"\n✅  Saved inputs & outputs to CSV: {csv_path}")


def load_onnx_and_run(onnx_path: str, out_file: str = None, *, enable_mem_profile: bool = True,):
    """
    Load an ONNX model, generate random inputs, run inference,
    print details and persist I/O tensors to a CSV file.

    Parameters
    ----------
    onnx_path : str
        Path to the ``*.onnx`` model.
    out_file : str | None
        Destination CSV file.  If ``None`` a file named ``<model_name>.csv``
        in the same directory as the model is created.
    enable_mem_profile : bool
        If True, ONNX‑Runtime profiling is turned on and the function
        prints a table with the memory usage of each operator.
    """
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # --------------------------------------------------
    # 1️⃣ Load the model (optional – sanity check)
    # --------------------------------------------------
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"\n=== Model: {onnx_path} ===")
    print(f"IR version: {model.ir_version}, opset: {model.opset_import[0].version}")

    # --------------------------------------------------
    # 2️⃣ Create an ONNX Runtime session
    # --------------------------------------------------
    sess_opts = ort.SessionOptions()
    if enable_mem_profile:
        # This flag creates a <model_name>_profile.json file after the first run
        sess_opts.enable_profiling = True
        sess_opts.add_session_config_entry("session.enable_mem_profile", "1")
    sess = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])
    # --------------------------------------------------
    # 3️⃣ Build random inputs
    # --------------------------------------------------
    feed = {}
    print("\n--- Model Inputs ---")
    for i, meta in enumerate(sess.get_inputs()):
        name = meta.name
        shape = meta.shape
        onnx_type = meta.type
        np_dtype = ORT_TYPE_TO_NP.get(onnx_type)

        if np_dtype is None:
            raise RuntimeError(f"Unsupported ONNX type '{onnx_type}' for input '{name}'")

        arr = _make_random_input(shape, np_dtype)
        feed[name] = arr

        print(f"{i+1}. name : {name}")
        print(f"   shape: {shape}  -> concrete: {arr.shape}")
        print(f"   type : {onnx_type} -> numpy dtype: {np_dtype}")

    # --------------------------------------------------
    # 4️⃣ Run inference
    # --------------------------------------------------
    print("\n--- Running inference ---")
    outputs = sess.run(None, feed)   # ``None`` → return all model outputs

    if enable_mem_profile:
        # The profiling file is written the first time ``run`` is called.
        prof_file = sess.end_profiling()
        try:
            with open(prof_file, "r") as f:
                prof_json = json.load(f)
        except Exception as exc:
            print(f"⚠️  Could not read profiling file: {exc}")
            prof_json = []

        # Build a nice table:  Node name | Op type | Memory (KB)
        from tabulate import tabulate

        rows = []
        peak_mem_overall = 0
        for entry in prof_json:
            # ``mem_usage`` is in bytes; not all entries have it (e.g. kernel launches)
            mem_bytes = entry.get("mem_usage")
            if mem_bytes is None:
                continue
            node_name = entry.get("node_name", "N/A")
            op_type = entry.get("op_name", "N/A")
            rows.append([node_name, op_type, f"{mem_bytes / 1024:.1f} KB"])
            peak_mem_overall = max(peak_mem_overall, mem_bytes)

        if rows:
            print("\n=== Per‑layer peak memory usage (from ONNX‑Runtime profiler) ===")
            print(tabulate(rows, headers=["Node", "Op", "Peak Mem (KB)"], tablefmt="github"))
            print(f"\nOverall peak memory for this inference: {peak_mem_overall / 1024:.1f} KB")
        else:
            print("⚠️  Profiling data did not contain any memory‑usage entries.")
    # --------------------------------------------------
    # 5️⃣ Print outputs
    # --------------------------------------------------
    out_meta = sess.get_outputs()
    print("\n--- Model Outputs ---")
    for idx, (meta, val) in enumerate(zip(out_meta, outputs)):
        print(f"{idx + 1}. name : {meta.name}")
        print(f"   shape: {meta.shape}")
        print(f"   type : {meta.type}")
        print(f"   value:\n{val}\n")
    # --------------------------------------------------
    # 5️⃣  OPTIONAL:  Report per‑layer memory usage
    # --------------------------------------------------
    # --------------------------------------------------
    # 6️⃣ Determine output CSV path (default = <_name>.csv)
    # --------------------------------------------------
    if out_file is None:
        base_dir, model_file = os.path.split(onnx_path)
        model_name, _ = os.path.splitext(model_file)
        out_file = os.path.join(base_dir, f"{model_name}.csv")

    # --------------------------------------------------
    # 7️⃣ Persist I/O tensors to CSV
    # --------------------------------------------------
    _save_io_csv(feed, outputs, out_file)

    return outputs


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CLI usage:
    #   python infer_with_onnx.py <model.onnx> [--out-file <csv_path>]
    # ------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python infer_with_onnx.py <model.onnx> [--out-file <csv_path>]")
        sys.exit(1)

    model_path = sys.argv[1]
    # Default: None → script will create "<model_name>_info.txt" next to the model
    info_file = None
    if "--info-file" in sys.argv:
        idx = sys.argv.index("--info-file")
        if idx + 1 >= len(sys.argv):
            raise ValueError("Missing argument after '--info-file'")
        info_file = sys.argv[idx + 1]
    dump_onnx_model_info(model_path, info_path=info_file)

    # Default: None → script will create "<model_name>.csv" next to the model
    csv_path = None

    # Simple optional flag parsing
    if "--out-file" in sys.argv:
        idx = sys.argv.index("--out-file")
        if idx + 3 >= len(sys.argv):
            raise ValueError("Missing argument after '--out-file'")
        csv_path = sys.argv[idx + 3]

    load_onnx_and_run(model_path, out_file=csv_path)