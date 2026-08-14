# Harden Compile-Hardening Follow-ups Implementation Plan

**Goal:** Fix three issues an independent peer code review found in the compile-hardening plan's aftermath: a live crash reachable through a supported radar/timeseries config, three regression tests too weak to catch the failures they're meant to guard, and a DataLoader resource leak newly reachable now that `radar_classification/train.py`'s `main()` is the live entrypoint.

**Architecture:** Three independent tasks, each touching different files with no shared state:
1. A one-condition change in two files (`radar_classification/train.py`, `timeseries_classification/train.py`) — fall back to loading datasets fresh when the quant-reuse cache is empty, instead of crashing.
2. Replace three source-text-grep tests with real behavioral tests that patch `compile_model_if_enabled` directly and assert on the call, mirroring an existing pattern already in this test suite (`test_anomalydetection_train_device_crash.py`).
3. Wrap radar's `main()` training body in `try/finally: shutdown_data_loaders(...)`, matching `image_classification/train.py`'s existing pattern exactly.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, pytest

## Global Constraints

- Python `==3.10.*`
- No new dependencies
- Task 1's fix must preserve the fast path (reuse cached dataset within the same process during a float-then-quant sequence) — only the "cache is empty" case should trigger a fresh load
- Task 2's new tests must genuinely fail if: the `compile_model_if_enabled` call is deleted, its return value is not assigned back to `model`, it's moved to the wrong position relative to `move_model_to_device`/`setup_distributed_model`, or it's placed in an unreachable branch
- Task 3 must not change any other structural aspect of radar's `main()` — pure re-indentation plus the try/finally wrapper and the one new call

## Context: Why This Is Needed

An independent peer code review of the compile-hardening plan (`docs/superpowers/plans/2026-08-13-compile-hardening-radar-image-audio.md`) found three issues in code that plan touched or made newly reachable:

**1. `run_quant_train_only: True` crashes for radar (and identically for `timeseries_classification`).** `main()`'s dataset loading (`radar_classification/train.py:193-200`) only calls `load_datasets(...)` when `args.quantization` is falsy; when true, it reads from the module-global `dataset_load_state` cache, populated only by a prior float-training call **in the same process**. `radar_base.py:401-426` has a documented, supported config (`run_quant_train_only`) that skips the float pass and calls `train_module.run(args)` once with `--quantization` set — so `dataset_load_state['dataset']` is still `None`, and `num_classes = len(dataset.classes)` (line 217) raises `AttributeError: 'NoneType' object has no attribute 'classes'`. Under `main_debug` (the function `main()` replaced) this config completed without crashing — `main_debug` loaded datasets unconditionally — but silently produced a non-quantized float model in the quantized output path, which is also wrong. The regression is "silently wrong" becoming "crashes loudly"; the fix here makes it neither.

**2. Three of five new `compile_model_if_enabled` regression tests are source-text greps** (`tests/test_radar_compile_wired.py`, `tests/test_image_classification_compile_wired.py`, `tests/test_audio_classification_compile_wired.py`) — `inspect.getsource(main)` + `"compile_model_if_enabled(" in source`. This passes even if: the call's return value isn't assigned back to `model`; the call moves after `setup_distributed_model` or the `NeuralNetworkWithPreprocess` wrap (wrong shape); the call sits in a dead branch; or the string appears in a comment. This repo already has a stronger pattern for exactly this kind of assertion — `tinyml-tinyverse/tests/test_anomalydetection_train_device_crash.py`'s third test drives the real `main()` with its dependency chain mocked and asserts on what a specific downstream call received. `tinyml-tinyverse/tests/test_train_best_epoch_bugs_vision_audio.py` already has working, complete mock harnesses for `image_classification.train.main()` and `audio_classification.train.main()` specifically — reuse those as the base rather than building new ones.

**3. `radar_classification/train.py`'s `main()` never calls `shutdown_data_loaders`.** `create_data_loaders` (`train_base.py:995-1000`) sets `persistent_workers=True` whenever `args.workers > 0` (default 8). `image_classification/train.py` (and all four `timeseries_*` scripts) wrap their training body in `try: ... finally: shutdown_data_loaders(data_loader, data_loader_test)` (see `image_classification/train.py:297,472-473`) specifically to clean these up. Radar's `main()` has no such wrapping at all. This was moot while `main_debug` (which built loaders without `persistent_workers`) was the live entrypoint; it's a real leak now that `main()` is live.

---

## File Map

| Action | Path | Responsibility |
|--------|------|-----------------|
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py:194` | Fall back to fresh load when quant-reuse cache is empty |
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_classification/train.py:214` | Same fix |
| Modify | `tinyml-tinyverse/tests/test_radar_compile_wired.py` | Replace source-grep with real behavioral test |
| Modify | `tinyml-tinyverse/tests/test_image_classification_compile_wired.py` | Same |
| Modify | `tinyml-tinyverse/tests/test_audio_classification_compile_wired.py` | Same |
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py` | Wrap `main()`'s training body in `try/finally: shutdown_data_loaders(...)` |

---

## Task 1: Fix the `run_quant_train_only` crash in radar and timeseries_classification

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py` (lines 193-200)
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_classification/train.py` (lines 213-220)

**Interfaces:**
- Consumes: `load_datasets` (already imported in both files, unchanged signature)

- [x] **Step 1: Write the failing test**

```python
"""Regression test: radar_classification.train.main() must not crash when
args.quantization is set and dataset_load_state's cache is empty (the
run_quant_train_only config: a standalone quantized run with no preceding
float-training call in the same process to populate the cache)."""
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import torch

from tinyml_tinyverse.references.radar_classification import train as radar_train


class _FakeDataset:
    classes = ['a', 'b']
    inverse_label_map = {0: 'a', 1: 'b'}
    X = torch.zeros((4, 8))
    Y = torch.zeros((4,), dtype=torch.long)

    def __getitem__(self, i):
        return self.X[i], self.X[i], self.Y[i]

    def __len__(self):
        return 4


def test_main_does_not_crash_when_quantization_cache_is_empty():
    radar_train.dataset_load_state['dataset'] = None
    radar_train.dataset_load_state['dataset_test'] = None
    radar_train.dataset_load_state['train_sampler'] = None
    radar_train.dataset_load_state['test_sampler'] = None

    args = Namespace(
        quantization=True, data_path='/fake', output_dir='/tmp/fake-radar-quant-only',
        gof_test=False, frame_size='None', dont_train_just_feat_ext='False',
        load_saved_model='None', nas_enabled='False', generic_model=True,
        model='LINEAR_4L_PC', model_config=None, model_spec=None, dual_op=False,
        output_int=True, quantization_method='QAT', weight_bitwidth=8,
        activation_bitwidth=8, epochs=1, start_epoch=0, label_smoothing=0.0,
        distributed=False, apex=False, print_freq=10, opset_version=17,
        gen_golden_vectors=False, DEBUG=False,
    )

    fake_dataset = _FakeDataset()
    fake_loaders = ([1], [1])

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            radar_train, "setup_training_environment",
            return_value=(radar_train.getLogger("test"), torch.device("cpu"))))
        stack.enter_context(patch.object(radar_train, "prepare_transforms"))
        stack.enter_context(patch.object(
            radar_train, "load_datasets",
            return_value=(fake_dataset, fake_dataset, None, None)))
        stack.enter_context(patch.object(radar_train, "create_data_loaders", return_value=fake_loaders))
        stack.enter_context(patch.object(radar_train.models, "get_model", return_value=torch.nn.Linear(8, 2)))
        stack.enter_context(patch.object(radar_train, "log_model_summary"))
        stack.enter_context(patch.object(radar_train, "load_pretrained_weights", side_effect=lambda m, a, l: m))
        stack.enter_context(patch.object(radar_train, "handle_export_only", return_value=False))
        stack.enter_context(patch.object(radar_train, "move_model_to_device"))
        stack.enter_context(patch.object(radar_train, "compile_model_if_enabled", side_effect=lambda m, a, l, **kw: m))
        stack.enter_context(patch.object(
            radar_train, "setup_distributed_model", side_effect=lambda m, a, d: (m, m, None)))
        stack.enter_context(patch.object(
            radar_train, "setup_optimizer_and_scheduler", return_value=(MagicMock(), MagicMock())))
        stack.enter_context(patch.object(radar_train, "resume_from_checkpoint"))
        stack.enter_context(patch.object(radar_train.utils, "quantization_wrapped_model", side_effect=lambda m, *a, **kw: m))
        stack.enter_context(patch.object(radar_train.utils, "train_one_epoch_classification"))
        stack.enter_context(patch.object(
            radar_train.utils, "evaluate_classification",
            return_value=(1.0, 1.0, 1.0, {}, [], [])))
        stack.enter_context(patch.object(radar_train, "save_checkpoint"))
        stack.enter_context(patch.object(radar_train.utils, "save_on_master"))
        stack.enter_context(patch.object(radar_train.utils, "print_file_level_classification_summary"))
        stack.enter_context(patch.object(radar_train.utils, "export_model"))
        stack.enter_context(patch.object(radar_train, "log_training_time"))

        # Should not raise. Pre-fix: AttributeError on dataset_load_state['dataset'] is None.
        radar_train.main(0, args)
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_radar_quant_only_no_crash.py -v`
Expected: FAIL with `AttributeError: 'NoneType' object has no attribute 'classes'`

- [x] **Step 3: Fix radar_classification/train.py**

Change:
```python
    if args.quantization:
        dataset, dataset_test, train_sampler, test_sampler = (dataset_load_state['dataset'], dataset_load_state['dataset_test'],
                                                               dataset_load_state['train_sampler'], dataset_load_state['test_sampler'])
    else:
```
to:
```python
    if args.quantization and dataset_load_state['dataset'] is not None:
        dataset, dataset_test, train_sampler, test_sampler = (dataset_load_state['dataset'], dataset_load_state['dataset_test'],
                                                               dataset_load_state['train_sampler'], dataset_load_state['test_sampler'])
    else:
```

- [x] **Step 4: Run test to verify it passes**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_radar_quant_only_no_crash.py -v`
Expected: PASS

- [x] **Step 5: Apply the identical fix to timeseries_classification/train.py**

Same one-word condition change at line 214 (`if args.quantization:` → `if args.quantization and dataset_load_state['dataset'] is not None:`). Write an analogous test (`tests/test_timeseries_classification_quant_only_no_crash.py`), adapting the mock args/dataset shape to `timeseries_classification`'s `main()` signature (check its existing tests, e.g. any in `tests/` already driving this `main()`, for the right fake-dataset shape before writing from scratch).

- [x] **Step 6: Run both new tests plus the full suite**

Run: `cd tinyml-tinyverse && python -m pytest tests/ -v`
Expected: both new tests pass; the one pre-existing unrelated failure (`test_anomalydetection_train_device_crash.py::test_main_passes_a_torch_device_not_the_raw_args_device_string`, broken since PR #22, unrelated to this change) is the only failure, if any

- [x] **Step 7: Commit**

```bash
cd tinyml-tinyverse
git add tinyml_tinyverse/references/radar_classification/train.py tinyml_tinyverse/references/timeseries_classification/train.py tests/test_radar_quant_only_no_crash.py tests/test_timeseries_classification_quant_only_no_crash.py
git commit -m "fix: run_quant_train_only crashed when the dataset-reuse cache was empty"
```

---

## Task 2: Strengthen the three weak compile_model_if_enabled tests

**Files:**
- Modify: `tinyml-tinyverse/tests/test_radar_compile_wired.py`
- Modify: `tinyml-tinyverse/tests/test_image_classification_compile_wired.py`
- Modify: `tinyml-tinyverse/tests/test_audio_classification_compile_wired.py`

**Interfaces:**
- Consumes: `tinyml_tinyverse.references.common.train_base.compile_model_if_enabled` (patch target), each module's own `main()`

- [x] **Step 1: Read the two reference patterns before writing anything**

`tinyml-tinyverse/tests/test_anomalydetection_train_device_crash.py` (specifically its third test, the one driving `anomaly_train.main(0, args)` with the full dependency chain mocked) is the template for "drive the real `main()`, assert on what a specific patched call received." `tinyml-tinyverse/tests/test_train_best_epoch_bugs_vision_audio.py` already contains complete, working mock harnesses for `image_classification.train.main()` and `audio_classification.train.main()` — reuse those fixtures/mocking setups rather than rebuilding them; only the specific assertion (on `compile_model_if_enabled`) is new.

- [x] **Step 2: Rewrite test_radar_compile_wired.py**

Replace the `inspect.getsource` substring check with a test that patches `compile_model_if_enabled` on the `radar_train` module, drives `main(0, args)` with everything else mocked (adapt the mock harness from Task 1's new radar test above, which already has a complete working mock set for `main()` — reuse it directly), and asserts:
- `compile_model_if_enabled` was called exactly once
- with `model` as the first positional arg (the pre-`setup_distributed_model`, pre-`NeuralNetworkWithPreprocess`-wrap model)
- with `input_shape=(1,) + dataset.X.shape[1:]` (assert the actual kwarg value, not just its presence)
- that its return value was what got passed into `setup_distributed_model` (i.e., the compiled/wrapped model is what continues through the rest of `main()`, not silently discarded)

- [x] **Step 3: Run it, verify RED then GREEN**

Temporarily verify this test fails against the *pre-Task-1-of-the-original-compile-hardening-plan* code shape (e.g. by checking it would fail if the `model = compile_model_if_enabled(...)` line were reverted to not exist, or not reassign `model`) — the simplest way is to comment out the assignment locally, confirm the new test fails, then restore it and confirm it passes. Document this RED/GREEN evidence in the report even though the underlying fix already shipped weeks ago; the point is proving the *new test* has teeth, not re-doing the original fix.

- [x] **Step 4: Repeat Steps 2-3 for test_image_classification_compile_wired.py**

Adapt using `test_train_best_epoch_bugs_vision_audio.py`'s existing `image_classification.train.main()` mock harness as the base.

- [x] **Step 5: Repeat Steps 2-3 for test_audio_classification_compile_wired.py**

Adapt using `test_train_best_epoch_bugs_vision_audio.py`'s existing `audio_classification.train.main()` mock harness as the base.

- [x] **Step 6: Run the full suite**

Run: `cd tinyml-tinyverse && python -m pytest tests/ -v`
Expected: all pass except the one known pre-existing unrelated failure

- [x] **Step 7: Commit**

```bash
cd tinyml-tinyverse
git add tests/test_radar_compile_wired.py tests/test_image_classification_compile_wired.py tests/test_audio_classification_compile_wired.py
git commit -m "test: replace source-text compile_model_if_enabled checks with real behavioral assertions"
```

---

## Task 3: Fix radar's missing shutdown_data_loaders (DataLoader worker leak)

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py` (imports ~line 65-89; `main()` body from the line after `create_data_loaders` at line 223 through the end of the function, currently ~line 336)

**Interfaces:**
- Consumes: `shutdown_data_loaders` from `..common.train_base` (already used by `image_classification/train.py` and the four `timeseries_*` scripts — not yet imported in radar's file)

- [x] **Step 1: Write a test that verifies shutdown_data_loaders is called**

```python
"""Regression test: radar_classification.train.main() must call
shutdown_data_loaders() before returning, so persistent DataLoader worker
processes (enabled whenever --workers > 0, the default) are cleaned up.
image_classification and all four timeseries_* reference scripts already
do this; radar's main() did not."""
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import torch

from tinyml_tinyverse.references.radar_classification import train as radar_train


class _FakeDataset:
    classes = ['a', 'b']
    inverse_label_map = {0: 'a', 1: 'b'}
    X = torch.zeros((4, 8))
    Y = torch.zeros((4,), dtype=torch.long)

    def __getitem__(self, i):
        return self.X[i], self.X[i], self.Y[i]

    def __len__(self):
        return 4


def test_main_calls_shutdown_data_loaders():
    args = Namespace(
        quantization=False, data_path='/fake', output_dir='/tmp/fake-radar-shutdown',
        gof_test=False, frame_size='None', dont_train_just_feat_ext='False',
        load_saved_model='None', nas_enabled='False', generic_model=True,
        model='LINEAR_4L_PC', model_config=None, model_spec=None, dual_op=False,
        output_int=True, quantization_method='QAT', weight_bitwidth=8,
        activation_bitwidth=8, epochs=1, start_epoch=0, label_smoothing=0.0,
        distributed=False, apex=False, print_freq=10, opset_version=17,
        gen_golden_vectors=False, DEBUG=False,
    )
    fake_dataset = _FakeDataset()
    fake_loaders = ([1], [1])

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            radar_train, "setup_training_environment",
            return_value=(radar_train.getLogger("test"), torch.device("cpu"))))
        stack.enter_context(patch.object(radar_train, "prepare_transforms"))
        stack.enter_context(patch.object(
            radar_train, "load_datasets",
            return_value=(fake_dataset, fake_dataset, None, None)))
        stack.enter_context(patch.object(radar_train, "create_data_loaders", return_value=fake_loaders))
        stack.enter_context(patch.object(radar_train.models, "get_model", return_value=torch.nn.Linear(8, 2)))
        stack.enter_context(patch.object(radar_train, "log_model_summary"))
        stack.enter_context(patch.object(radar_train, "load_pretrained_weights", side_effect=lambda m, a, l: m))
        stack.enter_context(patch.object(radar_train, "handle_export_only", return_value=False))
        stack.enter_context(patch.object(radar_train, "move_model_to_device"))
        stack.enter_context(patch.object(radar_train, "compile_model_if_enabled", side_effect=lambda m, a, l, **kw: m))
        stack.enter_context(patch.object(
            radar_train, "setup_distributed_model", side_effect=lambda m, a, d: (m, m, None)))
        stack.enter_context(patch.object(
            radar_train, "setup_optimizer_and_scheduler", return_value=(MagicMock(), MagicMock())))
        stack.enter_context(patch.object(radar_train, "resume_from_checkpoint"))
        stack.enter_context(patch.object(radar_train.utils, "quantization_wrapped_model", side_effect=lambda m, *a, **kw: m))
        stack.enter_context(patch.object(radar_train.utils, "train_one_epoch_classification"))
        stack.enter_context(patch.object(
            radar_train.utils, "evaluate_classification",
            return_value=(1.0, 1.0, 1.0, {}, [], [])))
        stack.enter_context(patch.object(radar_train, "save_checkpoint"))
        stack.enter_context(patch.object(radar_train.utils, "save_on_master"))
        stack.enter_context(patch.object(radar_train.utils, "print_file_level_classification_summary"))
        stack.enter_context(patch.object(radar_train.utils, "export_model"))
        stack.enter_context(patch.object(radar_train, "log_training_time"))
        mock_shutdown = stack.enter_context(patch.object(radar_train, "shutdown_data_loaders"))

        radar_train.main(0, args)

    mock_shutdown.assert_called_once_with(*fake_loaders)
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_radar_shutdown_data_loaders.py -v`
Expected: FAIL — `shutdown_data_loaders` is never imported or called, so patching it and asserting `assert_called_once_with` fails (or the import patch itself fails since the name doesn't exist yet — either failure mode confirms the gap)

- [x] **Step 3: Add the import**

Add `shutdown_data_loaders` to the existing `from ..common.train_base import (...)` block in `radar_classification/train.py`.

- [x] **Step 4: Wrap the training body in try/finally**

Change the structure from (data loaders created, then everything else at the same indentation level through end of function) to:
```python
    data_loader, data_loader_test = create_data_loaders(dataset, dataset_test, train_sampler, test_sampler, args, gpu)

    try:
        logger.info("Creating model")
        # ... (everything currently between here and the end of main(), re-indented one level deeper)
    finally:
        shutdown_data_loaders(data_loader, data_loader_test)
```
Match `image_classification/train.py:297-473`'s exact wrapping boundaries and style — the `try:` opens right after `create_data_loaders`, the `finally:` closes right before the function ends (after the `gen_golden_vectors` block, which is the last statement in `main()`).

- [x] **Step 5: Run test to verify it passes**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_radar_shutdown_data_loaders.py -v`
Expected: PASS

- [x] **Step 6: Run the full suite plus a real (non-mocked) manual check**

Run: `cd tinyml-tinyverse && python -m pytest tests/ -v` — expect only the known pre-existing unrelated failure.

Then run an actual training pass on the synthetic radar fixture (reuse `make_radar_fixture.py`/the argv pattern from `docs/superpowers/plans/2026-08-13-radar-training-entrypoint-fix.md`'s Task 1) for a couple of epochs, `--device cpu`, to confirm the re-indented function still runs correctly end to end (the re-indentation is mechanical but touches ~110 lines — a real run is the best check that nothing was mis-indented into the wrong scope).

- [x] **Step 7: Commit**

```bash
cd tinyml-tinyverse
git add tinyml_tinyverse/references/radar_classification/train.py tests/test_radar_shutdown_data_loaders.py
git commit -m "fix: radar_classification main() leaked persistent DataLoader workers"
```
