# Radar Training Entrypoint Fix Implementation Plan

**Goal:** Wire the radar classification training script's `run()` entrypoint to the fully-featured `main()` function instead of the leftover `main_debug()` harness, then confirm on real hardware that this closes the MPS-vs-CPU performance gap it currently causes.

**Architecture:** `run_distributed(main_debug, args)` becomes `run_distributed(main, args)` in `radar_classification/train.py`. `main()` already exists and is fully wired to the shared, hardened training infrastructure (`quantization_wrapped_model`, `compile_model_if_enabled`/`apply_hardware_defaults`, `resume_from_checkpoint`, `create_data_loaders`) — it is simply never called today. No new code is needed for the fix itself; the risk is regression in radar-specific behavior that only `main_debug` currently exercises, which Task 1's test and manual run guard against.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, pytest

> **Correction (see ## Results below):** the Goal and Architecture above assumed `main()` was already wired to `compile_model_if_enabled`/`apply_hardware_defaults`. It is not. Task 1's fix is real (quantization and checkpoint-resume now work) but does **not** close the MPS-vs-CPU gap — see Results for the actual post-fix numbers and root cause.

## Global Constraints

- Python `==3.10.*`
- No new dependencies
- `main_debug()` itself is left in place (unreferenced) — it's the author's own notebook-parity debug harness per its docstring, not this plan's concern to delete
- The fix must not change CLI argument names or defaults — only which internal function `run()` dispatches to

## Context: Why This Is Needed

`BaseRadarModelTraining.run()` (`tinyml-modelmaker/tinyml_modelmaker/ai_modules/radar/training/tinyml_tinyverse/radar_base.py`) calls `self.train_module.run(args)` for both the float-training pass and the quantized-training pass. `train_module` is wired to `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py`, whose `run()` is:

```python
def run(args):
    """Run training with optional distributed mode."""
    run_distributed(main_debug, args)
```

`main_debug()` is a hand-rolled duplicate of `main()` (its own comment: *"Following as close as possible steps from jupyter notebook to test if model learning plateau is coming from training loop"*). Confirmed by reading both functions side by side:

- `main_debug()` hardcodes `phase = 'FloatTrain'` unconditionally and never calls `quantization_wrapped_model` — the "QuantTrain" argv pass `radar_base.py` builds produces a second float-training run instead, logged under the wrong phase label, so `get_radar_classification_log_summary_regex()`'s `QuantTrain`-specific regexes never match anything.
- `main_debug()` never calls `compile_model_if_enabled` or `apply_hardware_defaults` (PRs #21–23) — no AMP, no `torch.compile`, regardless of device or hardware.
- `main_debug()` never calls `resume_from_checkpoint`.

Empirically confirmed this has a real, measurable performance cost, not just a theoretical one. Benchmarked `LINEAR_4L_PC` (the one registered radar model) on a synthetic 5-class radar-shaped fixture, 30 epochs, via the actual `radar_classification.train.run(args)` entrypoint as currently shipped:

| Device | Time/epoch |
|---|---|
| CPU | 0.602s |
| MPS | 1.145s (**1.90x slower than CPU**) |

MPS being slower than CPU is the expected signature of a small-op-heavy graph with no kernel fusion — exactly what `apply_hardware_defaults`/`compile_model_if_enabled` exists to fix elsewhere in the codebase. Task 2 re-runs this same benchmark once `main()` is live, to confirm and quantify the improvement.

---

## File Map

| Action | Path | Responsibility |
|--------|------|-----------------|
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py:522-524` | `run()` dispatches to `main`, not `main_debug` |
| Create | `tinyml-tinyverse/tests/test_radar_entrypoint_uses_main.py` | Regression test pinning the dispatch target |

---

## Task 1: Fix `run()` dispatch target, with regression test

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py` (function `run`, currently lines 522-524)
- Create: `tinyml-tinyverse/tests/test_radar_entrypoint_uses_main.py`

**Interfaces:**
- Consumes: `tinyml_tinyverse.references.radar_classification.train.main`, `.main_debug`, `.run`, `.run_distributed` (all already defined in the module — no new interfaces)

- [x] **Step 1: Write the failing test**

```python
"""Regression test: radar_classification.train.run() must dispatch to main(),
not main_debug() (a leftover notebook-parity harness that silently skips
quantization, AMP, and torch.compile for every radar training run)."""
from unittest.mock import patch

from tinyml_tinyverse.references.radar_classification import train as radar_train


def test_run_dispatches_to_main_not_main_debug():
    with patch.object(radar_train, "run_distributed") as mock_run_distributed:
        fake_args = object()
        radar_train.run(fake_args)

    mock_run_distributed.assert_called_once()
    dispatched_fn, dispatched_args = mock_run_distributed.call_args[0]

    assert dispatched_fn is radar_train.main, (
        f"run() dispatched to {dispatched_fn.__name__!r}, expected 'main'. "
        "main_debug() never applies quantization_wrapped_model, "
        "compile_model_if_enabled/apply_hardware_defaults, or "
        "resume_from_checkpoint -- wiring run() to it silently drops quantized "
        "training and hardware acceleration for every radar run."
    )
    assert dispatched_args is fake_args
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_radar_entrypoint_uses_main.py -v`
Expected: FAIL — `dispatched_fn` is `main_debug`, not `main`

- [x] **Step 3: Fix the dispatch target**

In `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py`, change:

```python
def run(args):
    """Run training with optional distributed mode."""
    run_distributed(main_debug, args)
```

to:

```python
def run(args):
    """Run training with optional distributed mode."""
    run_distributed(main, args)
```

- [x] **Step 4: Run test to verify it passes**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_radar_entrypoint_uses_main.py -v`
Expected: PASS

- [x] **Step 5: Manual end-to-end sanity check**

`main()` has never actually executed for radar before — confirm it runs clean on a real (if synthetic) dataset before trusting the unit test alone. Build a small fixture (5 class dirs of CSVs under `<root>/classes/`, balanced `annotations/instances_{train,test,val}_list.txt`) and drive `train.get_args_parser().parse_args([...]); train.run(args)` directly with `--device cpu`, a few epochs, `--quantization 0`. Confirm it completes and exports `model.onnx` without error. Then repeat with `--quantization 1` (previously silently a no-op under `main_debug`) and confirm the log now shows `QuantTrain` phase entries and a second exported model under the quantization output dir.

- [x] **Step 6: Commit**

```bash
cd tinyml-tinyverse
git add tinyml_tinyverse/references/radar_classification/train.py tests/test_radar_entrypoint_uses_main.py
git commit -m "fix: radar_classification run() was dispatching to main_debug, not main"
```

---

## Task 2: Re-benchmark MPS vs CPU with the fix in place

**Files:**
- None modified — this is a measurement task using the fixture/driver already built during investigation (`bench_radar.py`, `make_radar_fixture.py` in the session scratchpad — recreate if not available)

**Interfaces:**
- Consumes: `radar_classification.train.run(args)` (now dispatching to `main`, per Task 1)

- [x] **Step 1: Re-run the same benchmark used to characterize the bug**

Using the same synthetic fixture (5 classes, balanced annotation lists, `LINEAR_4L_PC`, batch size 16, 30 epochs) and the same driver pattern that produced the pre-fix numbers (constructs `argv`, calls `train.get_args_parser().parse_args(argv)`, times `train.run(args)`), run once with `--device cpu` and once with `--device mps`.

Run: `python bench_radar.py cpu` then `python bench_radar.py mps` (with `PYTORCH_ENABLE_MPS_FALLBACK=1` set)
Expected: both complete without error; timing printed as `total=...s for N epochs -> ...s/epoch`

- [x] **Step 2: Record the comparison**

Append a short results table to this plan (or a sibling `RESULTS.md` next to it) with pre-fix vs post-fix CPU and MPS per-epoch timing, and note whether `apply_hardware_defaults` actually enabled `torch.compile`/AMP for this run (check the training log for the relevant INFO lines from `compile_model_if_enabled`/`apply_hardware_defaults`).

- [x] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-08-13-radar-training-entrypoint-fix.md
git commit -m "docs: record post-fix MPS vs CPU benchmark for radar training"
```

---

## Results

Same fixture (5-class synthetic radar data), same driver (`bench_radar.py`, `LINEAR_4L_PC`, batch size 16, 30 epochs), run through `radar_classification.train.run(args)` — which now dispatches to `main()` per Task 1's fix.

| Run | Device | Time/epoch | Total (30 epochs) | MPS vs CPU |
|---|---|---|---|---|
| Pre-fix (`main_debug`, historical) | CPU | 0.602s | — | — |
| Pre-fix (`main_debug`, historical) | MPS | 1.145s | — | **1.90x slower** |
| Post-fix (`main`) | CPU | 0.621s | 18.62s | — |
| Post-fix (`main`) | MPS | 1.066s | 31.99s | **1.72x slower** |

**The gap did not close.** MPS is still substantially slower than CPU after the fix — 1.72x, essentially the same order as the pre-fix 1.90x. The ~9% wobble between the two ratios is consistent with ordinary run-to-run variance (different process, thermal state, etc.), not a structural change.

**Why: `compile_model_if_enabled`/`apply_hardware_defaults` never engage for radar, in either `main()` or `main_debug()`.** Grepping both run logs (`radar_out_cpu/run.log`, `radar_out_mps/run.log`) for the exact strings `compile_model_if_enabled` emits on success (`"Compiling model with torch.compile"`), on quantization skip (`"compile_model is enabled but quantization is also enabled"`), and on failure (`"torch.compile failed"`) — zero matches in either log. Reading the code confirms why:

- `radar_classification/train.py`'s `main()` (lines 187-334) calls `quantization_wrapped_model` (line 269) and `resume_from_checkpoint` (line 257) — those parts of Task 1's fix are real and now active — but it never imports or calls `compile_model_if_enabled` at all. Contrast with `timeseries_classification/train.py`, `timeseries_forecasting/train.py`, `timeseries_anomalydetection/train.py`, and `timeseries_regression/train.py`, which all import it from `..common.train_base` and call it before their training loops. `image_classification/train.py` and `audio_classification/train.py` also never call it — this looks like a scope gap in PR #22 ("compile-hardening", commit `dc8eeb2`), which only touched the four `timeseries_*` scripts, not the `radar_classification`/`image_classification`/`audio_classification` scripts.
- `apply_hardware_defaults` isn't part of `tinyml-tinyverse` at all — it lives in `tinyml-modelmaker/tinyml_modelmaker/utils/hardware_defaults.py` and is wired only into `tinyml_modelmaker/ai_modules/timeseries/params.py` (commit `edbabba`, "wire apply_hardware_defaults into timeseries init_params"). It runs one layer up, in modelmaker's argv-construction step, before `train.py` ever sees `--compile-model`/`--native-amp`. `bench_radar.py` calls `radar_classification.train.get_args_parser().parse_args(argv)` directly, bypassing modelmaker entirely, so this layer was never in play for either the pre-fix or post-fix benchmark regardless.
- Net effect: `--compile-model` defaults to `0` and `--native-amp` defaults to `False` in `get_base_args_parser()` (`train_base.py:200,221`), nothing in the radar call path ever overrides them, and even if it did, `main()` has no code path that would act on them. torch.compile and AMP were off for all four runs (pre-fix CPU/MPS, post-fix CPU/MPS) — the comparison is apples-to-apples on that axis, just not for the reason the plan's premise expected.

**Bottom line:** Task 1's fix is still correct and worth having — it wires up quantization (`QuantTrain` phase now actually runs quantization-aware training instead of a mislabeled second float pass) and checkpoint resume, both real, previously-silent gaps. But it does not touch torch.compile/AMP for radar, because that wiring was never built for the radar script in the first place (unlike the `timeseries_*` family). The MPS-slower-than-CPU result — expected for a small-op-heavy graph with no kernel fusion — persists post-fix because the mechanism that would fix it (`compile_model_if_enabled`) is not in `radar_classification/train.py`'s call graph at all. Closing this gap would require a separate change: wiring `compile_model_if_enabled` (and optionally AMP via `get_amp_context`/`get_grad_scaler`) into `radar_classification/train.py::main()`, mirroring what the `timeseries_*` scripts already do — out of scope for this plan.
