# Extend compile_model_if_enabled to Radar, Image, and Audio Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `compile_model_if_enabled` (the `torch.compile` warmup-and-fallback helper added by PR #22, currently only called from the four `timeseries_*` reference scripts) into `radar_classification`, `image_classification`, and `audio_classification`'s `main()` functions, so all seven reference training scripts get the same hardware-acceleration path — and measure whether it actually helps each one.

**Architecture:** Each of the three target `main()` functions already has the exact structural shape `compile_model_if_enabled` expects: `move_model_to_device(model, device, logger)` immediately followed by `setup_distributed_model(model, args, device)`, with `dataset.X` (post feature-extraction) available in scope. The fix is the same one-line insertion in all three files, matching `timeseries_classification/train.py:275` verbatim:
```python
model = compile_model_if_enabled(model, args, logger, input_shape=(1,) + dataset.X.shape[1:])
```
placed between those two calls, plus adding `compile_model_if_enabled` to each file's import from `..common.train_base`. No changes to `compile_model_if_enabled` itself or to `train_base.py`.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, pytest

## Global Constraints

- Python `==3.10.*`
- No new dependencies
- Zero change to behavior when `--compile-model` is not set (the default, `0`) — `compile_model_if_enabled` already no-ops in that case; do not add any new conditionals around the call
- Do not touch `compile_model_if_enabled` itself, `train_base.py`, or any file outside the three target `train.py` files (+ their new/modified test files)
- Each task's benchmark must reuse the same measurement methodology as the radar entrypoint fix plan (`docs/superpowers/plans/2026-08-13-radar-training-entrypoint-fix.md`) — synthetic fixture, same device/epoch/batch-size reporting style — so results are comparable across all four now-benchmarked modules (timeseries already had compile; radar was benchmarked without it in that plan; this plan adds image and audio, and re-benchmarks radar with it now on)
- **Verification runs on two machines, not one:** local Mac (CPU vs MPS, `~/.venv-tinyml`) AND GX10 (CPU vs CUDA — `NVIDIA GB10`, `~/jupyterlab/.venv`, torch 2.9.0+cu130, reachable via `ssh -i ~/.ssh/gx10_key martin@gx10-singularity.skunk-mercat.ts.net`). Implementer subagents are not expected to have GX10 SSH access — the controller runs the GX10 leg directly after each task's local implementation is approved, and appends those results alongside the local ones in that task's Results section before considering the task's verification complete.

## Context: Why This Is Needed

While closing out the radar entrypoint fix (`docs/superpowers/plans/2026-08-13-radar-training-entrypoint-fix.md`), the whole-plan review found that fixing radar's `run()` dispatch (making `main()` live) did **not** close its MPS-vs-CPU performance gap, because `compile_model_if_enabled`/`apply_hardware_defaults` were never wired into radar's `main()` at all — only into the four `timeseries_*` scripts (PR #22, commit `dc8eeb2`). The same review flagged that `image_classification` and `audio_classification` have the identical gap:

```
grep -n "compile_model_if_enabled" tinyml-tinyverse/tinyml_tinyverse/references/image_classification/train.py
tinyml-tinyverse/tinyml_tinyverse/references/audio_classification/train.py
```
— zero matches in either file, confirmed directly against source, independent of the radar plan's own investigation.

Rather than fix radar alone and leave image/audio in the same state, this plan closes the gap everywhere it exists in one coordinated pass, so the three currently-uncompiled reference scripts converge on the same behavior as `timeseries_classification`, `timeseries_regression`, `timeseries_forecasting`, and `timeseries_anomalydetection` already have.

**Known unresolved question, not assumed:** whether `torch.compile` actually helps on hardware this small (radar's `LINEAR_4L_PC` is a 4-layer linear/BatchNorm model — the radar plan measured MPS at ~1.9x slower than CPU with no compile, and the theory is that fused kernels reduce per-op dispatch overhead, but this has not been confirmed for a model this size). Each task's benchmark step exists specifically to answer this per-module — report what you measure, not what the theory predicts.

---

## File Map

| Action | Path | Responsibility |
|--------|------|-----------------|
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py` | Wire `compile_model_if_enabled` into `main()` |
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/image_classification/train.py` | Wire `compile_model_if_enabled` into `main()` |
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/audio_classification/train.py` | Wire `compile_model_if_enabled` into `main()` |
| Create | `tinyml-tinyverse/tests/test_radar_compile_wired.py` | Regression test: radar's `main()` calls `compile_model_if_enabled` |
| Create | `tinyml-tinyverse/tests/test_image_classification_compile_wired.py` | Same, image classification |
| Create | `tinyml-tinyverse/tests/test_audio_classification_compile_wired.py` | Same, audio classification |

---

## Task 1: Wire `compile_model_if_enabled` into radar_classification, with test + benchmark

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py` (imports ~line 65-89; insertion point currently lines 251-254, between `move_model_to_device` and `setup_distributed_model`)
- Create: `tinyml-tinyverse/tests/test_radar_compile_wired.py`

**Interfaces:**
- Consumes: `tinyml_tinyverse.references.common.train_base.compile_model_if_enabled` (existing, unmodified — signature `compile_model_if_enabled(model, args, logger, input_shape=None)`, returns the (possibly compiled) model)
- Produces: nothing new consumed by later tasks — Tasks 2 and 3 are independent, same pattern applied to different files

- [ ] **Step 1: Write the failing test**

```python
"""Regression test: radar_classification.train.main() must call
compile_model_if_enabled, matching the pattern already used in the 4
timeseries_* reference scripts (PR #22). Without this, --compile-model is
silently a no-op for radar regardless of what the caller requests."""
import inspect

from tinyml_tinyverse.references.radar_classification import train as radar_train


def test_main_calls_compile_model_if_enabled():
    main_source = inspect.getsource(radar_train.main)
    assert "compile_model_if_enabled(" in main_source, (
        "main() does not call compile_model_if_enabled -- torch.compile/AMP "
        "hardware acceleration would silently not apply to radar training."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_radar_compile_wired.py -v`
Expected: FAIL — `"compile_model_if_enabled("` not present in `main`'s source

- [ ] **Step 3: Add the import and the call**

In `tinyml-tinyverse/tinyml_tinyverse/references/radar_classification/train.py`, add `compile_model_if_enabled` to the existing import from `..common.train_base` (around line 65-89 — find the multi-line `from ..common.train_base import (...)` block and add it as a new entry, alphabetically or grouped with `move_model_to_device`/`compile_model_if_enabled`-adjacent imports per the file's existing style).

Then in `main()`, change:
```python
    move_model_to_device(model, device, logger)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
```
to:
```python
    move_model_to_device(model, device, logger)
    model = compile_model_if_enabled(model, args, logger, input_shape=(1,) + dataset.X.shape[1:])
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_radar_compile_wired.py -v`
Expected: PASS

- [ ] **Step 5: Manual end-to-end sanity check with --compile-model 1**

Using the synthetic radar fixture from the entrypoint-fix plan (regenerate via `make_radar_fixture.py` if not present — see that plan's Task 1 for the recipe), drive `train.run(args)` directly with `--device cpu --compile-model 1`, a few epochs. Confirm it completes without error and the log shows `compile_model_if_enabled`'s own INFO lines (check `train_base.py`'s `compile_model_if_enabled` for the exact log message text first — grep the run log for it). Then confirm `--compile-model 0` (the default) still behaves identically to before this change — no compile-related log lines, same training behavior.

- [ ] **Step 6: Benchmark CPU vs MPS with compile now enabled**

Reuse the benchmark driver pattern from the entrypoint-fix plan (`bench_radar.py` in the session scratchpad), but add `--compile-model 1` to the argv. Run once on `cpu`, once on `mps` (`PYTORCH_ENABLE_MPS_FALLBACK=1`), same 30 epochs / batch size 16 / `LINEAR_4L_PC` as the entrypoint-fix plan's benchmark, so the numbers are directly comparable. Record: does compile change the CPU number, the MPS number, or the ratio between them, versus the entrypoint-fix plan's post-fix-no-compile numbers (CPU 0.621s/epoch, MPS 1.066s/epoch, 1.72x)? Report what you measure even if compile makes things worse or has no effect — that's a legitimate finding for a model this small, not a task failure.

- [ ] **Step 7: Record results and commit**

Append a `## Results (Task 1: radar)` section to this plan doc with the benchmark table and a short explanation.

```bash
cd tinyml-tinyverse
git add tinyml_tinyverse/references/radar_classification/train.py tests/test_radar_compile_wired.py
git add ../docs/superpowers/plans/2026-08-13-compile-hardening-radar-image-audio.md
git commit -m "feat: wire compile_model_if_enabled into radar_classification main()"
```

---

## Task 2: Wire `compile_model_if_enabled` into image_classification, with test + benchmark

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/image_classification/train.py` (imports ~line 92-111; insertion point currently lines 329-332, between `move_model_to_device` and `setup_distributed_model`)
- Create: `tinyml-tinyverse/tests/test_image_classification_compile_wired.py`

**Interfaces:**
- Consumes: same `compile_model_if_enabled` as Task 1 — independent of Task 1, do not wait for it or reuse its branch

- [ ] **Step 1: Write the failing test**

```python
"""Regression test: image_classification.train.main() must call
compile_model_if_enabled, matching the timeseries_* pattern (PR #22)."""
import inspect

from tinyml_tinyverse.references.image_classification import train as image_train


def test_main_calls_compile_model_if_enabled():
    main_source = inspect.getsource(image_train.main)
    assert "compile_model_if_enabled(" in main_source, (
        "main() does not call compile_model_if_enabled -- torch.compile/AMP "
        "hardware acceleration would silently not apply to image classification training."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_image_classification_compile_wired.py -v`
Expected: FAIL

- [ ] **Step 3: Add the import and the call**

Add `compile_model_if_enabled` to the existing `from ..common.train_base import (...)` block (~line 92-111). Then change:
```python
        move_model_to_device(model, device, logger)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
```
to:
```python
        move_model_to_device(model, device, logger)
        model = compile_model_if_enabled(model, args, logger, input_shape=(1,) + dataset.X.shape[1:])
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
```
**Note the indentation in this file is one level deeper than radar/audio (this block sits inside an outer block in image_classification) — match the surrounding indentation exactly, don't copy radar's column position verbatim.**

**Watch for:** this file has an `args.nn_for_feature_extraction` branch elsewhere (used at export time, line ~450-454) that chooses between `dataset.X_raw.shape` and `dataset.X.shape` depending on whether an NN is used for feature extraction. Confirm which shape the model actually consumes *at the point where you're inserting the compile call* (before or after any feature-extraction wrapping) — if `nn_for_feature_extraction` changes what the raw model's forward() expects at this point in `main()`, use the matching shape instead of assuming `dataset.X.shape` unconditionally. If unsure after reading the surrounding ~50 lines, ask before proceeding — this is exactly the kind of task-specific judgment call the brief can't resolve for you.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_image_classification_compile_wired.py -v`
Expected: PASS

- [ ] **Step 5: Manual end-to-end sanity check with --compile-model 1**

Build or reuse a small synthetic image classification fixture (check `tinyml-tinyverse/tests/` for an existing image dataset test fixture/helper before building one from scratch — this repo likely already has one given image_classification has existing tests). Drive `train.run(args)` with `--device cpu --compile-model 1`, a few epochs. Confirm completion and compile-related log lines present. Confirm `--compile-model 0` still behaves as before.

- [ ] **Step 6: Benchmark CPU vs MPS with compile enabled**

Same methodology as Task 1 Step 6, adapted to whichever image model this benchmark uses (pick the smallest/fastest registered image model available, to keep iteration time reasonable) and image_classification's own CLI args. This is the first CPU-vs-MPS benchmark for image classification in either plan — there's no prior "no-compile" baseline to compare against from the earlier work, so also run once with `--compile-model 0` on both devices first to get that baseline, then `--compile-model 1` to see the delta, in the same benchmark session.

- [ ] **Step 7: Record results and commit**

Append `## Results (Task 2: image_classification)` to this plan doc.

```bash
cd tinyml-tinyverse
git add tinyml_tinyverse/references/image_classification/train.py tests/test_image_classification_compile_wired.py
git add ../docs/superpowers/plans/2026-08-13-compile-hardening-radar-image-audio.md
git commit -m "feat: wire compile_model_if_enabled into image_classification main()"
```

---

## Task 3: Wire `compile_model_if_enabled` into audio_classification, with test + benchmark

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/audio_classification/train.py` (imports ~line 92-111; insertion point currently lines 311-313, between `move_model_to_device` and `setup_distributed_model`)
- Create: `tinyml-tinyverse/tests/test_audio_classification_compile_wired.py`

**Interfaces:**
- Consumes: same `compile_model_if_enabled` — independent of Tasks 1 and 2

- [ ] **Step 1: Write the failing test**

```python
"""Regression test: audio_classification.train.main() must call
compile_model_if_enabled, matching the timeseries_* pattern (PR #22)."""
import inspect

from tinyml_tinyverse.references.audio_classification import train as audio_train


def test_main_calls_compile_model_if_enabled():
    main_source = inspect.getsource(audio_train.main)
    assert "compile_model_if_enabled(" in main_source, (
        "main() does not call compile_model_if_enabled -- torch.compile/AMP "
        "hardware acceleration would silently not apply to audio classification training."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_audio_classification_compile_wired.py -v`
Expected: FAIL

- [ ] **Step 3: Add the import and the call**

Add `compile_model_if_enabled` to the existing `from ..common.train_base import (...)` block (~line 92-111). Then change:
```python
    move_model_to_device(model, device, logger)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
```
to:
```python
    move_model_to_device(model, device, logger)
    model = compile_model_if_enabled(model, args, logger, input_shape=(1,) + dataset.X.shape[1:])
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    model, model_without_ddp, model_ema = setup_distributed_model(model, args, device)
```

Same caveat as Task 2 applies here — audio_classification also has an `nn_for_feature_extraction` / `X_raw` vs `X` distinction at export time (line ~424-427). Check whether it affects what shape the model expects at this earlier insertion point before assuming `dataset.X.shape` unconditionally.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_audio_classification_compile_wired.py -v`
Expected: PASS

- [ ] **Step 5: Manual end-to-end sanity check with --compile-model 1**

Same approach as Task 2 Step 5, adapted to audio_classification's dataset/CLI. Check `tinyml-tinyverse/tests/` for existing audio fixture helpers first.

- [ ] **Step 6: Benchmark CPU vs MPS with compile enabled**

Same methodology as Task 2 Step 6 (baseline `--compile-model 0` then `--compile-model 1`, both devices), adapted to audio_classification's smallest registered model.

- [ ] **Step 7: Record results and commit**

Append `## Results (Task 3: audio_classification)` to this plan doc.

```bash
cd tinyml-tinyverse
git add tinyml_tinyverse/references/audio_classification/train.py tests/test_audio_classification_compile_wired.py
git add ../docs/superpowers/plans/2026-08-13-compile-hardening-radar-image-audio.md
git commit -m "feat: wire compile_model_if_enabled into audio_classification main()"
```

---

## Results (Task 1: radar)

**Wiring:** `compile_model_if_enabled` is now imported and called in `radar_classification/train.py`'s `main()`, between `move_model_to_device` and `setup_distributed_model`, exactly matching the `timeseries_*` pattern. Regression test `tests/test_radar_compile_wired.py` passes; `--compile-model 0` (default) produces zero compile-related log lines and identical behavior to before this change; `--compile-model 1` produces `compile_model_if_enabled`'s own log line (`Compiling model with torch.compile (backend=aot_eager)`) on both CPU and MPS, with no fallback warnings on either device — compilation and the warmup forward pass succeed cleanly for `LINEAR_4L_PC` on this hardware.

**Benchmark** — same fixture, `LINEAR_4L_PC`, batch size 16, 30 epochs, synthetic radar fixture, as the entrypoint-fix plan's post-fix-no-compile baseline:

| Config | Device | s/epoch | CPU/MPS ratio |
|---|---|---|---|
| No compile (entrypoint-fix plan baseline) | CPU | 0.621 | 1.72x |
| No compile (entrypoint-fix plan baseline) | MPS | 1.066 | |
| `--compile-model 1` (this task) | CPU | 0.765 | 1.70x |
| `--compile-model 1` (this task) | MPS | 1.303 | |

**Finding: `torch.compile` makes radar training slower on both devices, and does not close the CPU/MPS gap.** With compile enabled, CPU slows by ~23% (0.621s -> 0.765s/epoch) and MPS slows by ~22% (1.066s -> 1.303s/epoch). The CPU/MPS ratio is essentially unchanged (1.72x -> 1.70x) since both devices use the same `aot_eager` backend (the backend-selection logic in `compile_model_if_enabled` only routes CUDA to `inductor`; both CPU and MPS get `aot_eager`) and both slow down by a similar proportion.

This is consistent with `LINEAR_4L_PC` being a tiny 4-layer linear/BatchNorm model: `torch.compile`'s per-call dynamo tracing, guard-checking, and graph-dispatch overhead is fixed cost per training step, and for a model this small there isn't enough per-op dispatch overhead in the eager path for kernel fusion to recoup that cost. The theory in this plan's "Known unresolved question" (fused kernels reducing dispatch overhead) does not hold for this model size — compile is a net loss here, not neutral or beneficial. Wiring it in is still correct (it makes `--compile-model` functional, matching the other reference scripts, and lets a caller opt in for larger/more compute-bound models where the tradeoff may differ), but the flag should not be turned on by default for radar's small linear models based on this evidence.

**GX10 leg (controller-run, per Global Constraints):** same fixture/model/batch-size/epochs, `~/jupyterlab/.venv` (torch 2.9.0+cu130), `NVIDIA GB10`. GX10's clone was stale (`5035057`, pre-dating this whole plan) — updated to `origin/integration` (`e472749`) first. Environment needed several missing leaf packages installed (`--no-deps`, no `torch`/`torchvision` touched): `tabulate`, `torcheval`, `torchinfo`, `colorama`, `onnx`, `onnxruntime`, `protobuf`, `ml_dtypes`, `cryptography`, `PyWavelets`, `opencv-python`, `onnxscript`, `onnx_ir`. `cmsisdsp` and `torchaudio` were stubbed via `sys.modules` in the benchmark driver instead of installed for real — both are only needed transitively (by `timeseries_dataset.py`'s FFT path and `audio_dataset.py` respectively, eagerly imported by `datasets/__init__.py`) and never touched by radar's own code path; `cmsisdsp` in particular requires a multi-minute native CMSIS-DSP C build with no prebuilt ARM64 wheel, unrelated to what this benchmark measures.

| Config | Device | s/epoch |
|---|---|---|
| No compile | CPU | 4.093 |
| No compile | CUDA | 0.856 |
| `--compile-model 1` | CPU | 3.972 |
| `--compile-model 1` | CUDA | 0.943 |

CPU and CUDA aren't directly comparable to the Mac's CPU/MPS numbers (different, much less GX10-tuned CPU vs Apple Silicon CPU) — the useful comparison is each device against itself, with vs. without compile, on the same machine.

**Finding: on GX10, CUDA beats CPU by 4.8x even with no compile** (0.856s vs 4.093s/epoch) — a completely different picture from the Mac, where MPS lost to CPU. GX10's GPU has enough throughput advantage over its CPU that it wins decisively on this tiny model without any fusion help.

**Finding: `torch.compile` on CUDA never actually ran — it failed and silently fell back to eager.** The run log (`~/bench_radar/radar_out_cuda_compile1/run.log`) shows:
```
INFO: root.main: Compiling model with torch.compile (backend=inductor)
WARNING: root.main: torch.compile failed (or failed its warmup pass), falling back to eager mode:
CalledProcessError: Command '['/usr/bin/gcc', '.../cuda_utils.c', '-O3', '-shared', '-fPIC', ...
-lcuda', '-L.../triton/backends/nvidia/lib', ...]' returned non-zero exit status 1.
```
Triton's CUDA-kernel codegen fails to build `cuda_utils.c` via `gcc` on this box — a toolchain issue local to this environment (missing header/lib path for Triton's nvidia backend), not a `compile_model_if_enabled` defect. The `compile_model_if_enabled` warmup-and-fallback mechanism (built earlier this session, `docs/superpowers/plans/2026-07-28-compile-warmup-fallback.md`) caught the failure exactly as designed and fell back cleanly — training was not interrupted. This means the measured "compile=1, CUDA, 0.943s/epoch" number is **eager mode plus the one-time cost of a failed compile attempt**, not a real compiled-vs-uncompiled comparison; the CUDA compile question remains genuinely open on this hardware until the Triton/gcc toolchain issue is fixed here.

**Finding: CPU's `aot_eager` backend did engage successfully on GX10** (`INFO: root.main: Compiling model with torch.compile (backend=aot_eager)`, no fallback warning) and produced a small, real win (4.093 -> 3.972s/epoch, ~3%) — consistent in direction with the Mac's CPU-side aot_eager numbers being a similarly fixed-cost/small-model regime, though the Mac's own CPU result was a ~23% *loss*, not a ~3% gain — the two machines don't even agree on aot_eager's sign for this model, underscoring how workload-and-hardware-specific this tradeoff is.

**Net implication for the "should compile be on by default" question:** no clean "yes" or "no" emerges. Mac CPU/MPS: compile hurts on both. GX10 CPU: compile helps marginally. GX10 CUDA: unknown — the only device compile theoretically helps most (via `inductor`, not `aot_eager`) is the one device where it couldn't even run here. Recommend leaving `--compile-model` opt-in (its current default) rather than drawing a default-on conclusion from this data, and separately investigating the GX10 Triton/gcc build failure if CUDA compile behavior is worth knowing for real.

---

## Results (Task 2: image_classification)

**Wiring:** `compile_model_if_enabled` is now imported and called in `image_classification/train.py`'s `main()`, between `move_model_to_device` and `setup_distributed_model`, at the same indentation level as the surrounding `try:` block (one level deeper than radar/audio, per the brief's note). Regression test `tests/test_image_classification_compile_wired.py` passes; `--compile-model 0` (default) produces zero compile-related log lines and identical behavior to before this change; `--compile-model 1` produces `compile_model_if_enabled`'s own log line (`Compiling model with torch.compile (backend=aot_eager)`) on both CPU and MPS, with no fallback warnings on either device.

**X vs X_raw judgment call:** at the insertion point (right after `move_model_to_device`), `model` is still the plain model returned by `models.get_model(args.model, variables, num_classes, input_features=input_features, ...)` (or `torch.load` for `--load-saved-model`), constructed from `variables = dataset.X.shape[1]` / `input_features = dataset.X.shape[2]` a few lines earlier (line 291-292) — confirmed by the existing `torchinfo.summary(model, summary_input_shape)` call at line 320-322, which also uses `dataset.X.shape[1:]` for this same pre-wrap model. The `nn_for_feature_extraction` / `X_raw` vs `X` branch the brief flagged only appears later (lines 342-350), where `model` gets wrapped in `NeuralNetworkWithPreprocess` — that wrapping happens *after* our insertion point, not before it. So regardless of `args.nn_for_feature_extraction`, the raw model being compiled at this point in `main()` always consumes `dataset.X.shape[1:]`-shaped input; the brief's suggested `input_shape=(1,) + dataset.X.shape[1:]` is correct unconditionally, and no branching on `nn_for_feature_extraction` was needed.

**Fixture:** no ready-made image dataset fixture/driver existed in `tinyml-tinyverse/tests/` (the closest prior art, `tests/test_train_best_epoch_bugs_vision_audio.py`, fully mocks `get_model`/`create_data_loaders` and never drives real image data end-to-end). Built a small synthetic fixture instead: N class folders of 28x28 grayscale PNGs (`<root>/<class_name>/*.png`, matching `GenericImageDataset`'s fallback-discovery layout — no `annotations/*_list.txt` needed since `--dataset` stays at its `folder` default, not `modelmaker`), each class centered on a distinct mean pixel intensity plus noise. Model: `CNN_LENET5`, the smallest of the three registered image models (~a few hundred params vs. `CNN_IMG_MOBILENETV1/2_58K_NPU`'s ~58K), whose spec hard-codes a 28x28x1 input (`Linear(in_features=400, ...)` after two conv+pool blocks) — fixture images were generated at exactly that size. Driver script (`bench_image.py` in the session scratchpad) builds argv for `train.get_args_parser().parse_args(argv)` and times `train.run(args)`, same pattern as `bench_radar.py`.

**Manual E2E check (Step 5):** 3-class fixture (12 images/class), 2 epochs, batch size 8. `--device cpu --compile-model 0` (both pre- and post-code-change): completes cleanly, exports `model.onnx`, zero compile-related log lines. `--device cpu --compile-model 1`: completes cleanly, exports `model.onnx`, log shows `INFO: root.main: Compiling model with torch.compile (backend=aot_eager)` with no fallback warning.

**Benchmark (Step 6)** — 5-class fixture (20 images/class = 100 images), `CNN_LENET5`, batch size 16, 30 epochs, `--device cpu` / `--device mps` (`PYTORCH_ENABLE_MPS_FALLBACK=1`), each config run 3-4 times to check stability:

| Config | Device | s/epoch (steady-state avg) | compile engaged? |
|---|---|---|---|
| `--compile-model 0` (baseline) | CPU | 0.037 (n=3, range 0.037-0.037) | n/a (no compile lines in log) |
| `--compile-model 0` (baseline) | MPS | 0.073 (n=3 steady-state, range 0.069-0.077) | n/a (no compile lines in log) |
| `--compile-model 1` | CPU | 0.083 (n=3, range 0.083-0.084) | yes, `aot_eager`, no fallback |
| `--compile-model 1` | MPS | 0.094 (n=3 steady-state, range 0.092-0.096) | yes, `aot_eager`, no fallback |

**Finding: `torch.compile` makes image_classification training slower on both devices, same direction as radar, but with a smaller relative hit on MPS than on CPU.** CPU regresses ~124% (0.037s -> 0.083s/epoch) — proportionally worse than radar's ~23% CPU regression. MPS regresses only ~28% (0.073s -> 0.094s/epoch) — much milder than radar's ~22% MPS regression was *relatively* similar in magnitude, but here the CPU/MPS gap actually **narrows** with compile on: 0.073/0.037 = 1.98x (MPS slower, no compile) vs. 0.094/0.083 = 1.13x (MPS slower, with compile). CPU still wins in absolute terms either way, but compile shrinks rather than preserves the ratio — the opposite of radar, where the ratio stayed essentially flat (1.72x -> 1.70x). `CNN_LENET5` has real conv/pool ops (unlike radar's pure linear/BatchNorm stack), so there is at least some op-fusion opportunity for `aot_eager` to exploit on MPS, but for a model this tiny (~28x28 inputs, batch 16, a few hundred parameters) the fixed per-step dynamo tracing/guard overhead still dominates on both devices — it just dominates less on MPS relative to MPS's already-slower eager baseline.

**Methodology note — MPS cold-start artifact:** the very first MPS invocation of the whole benchmark session (`--compile-model 0`, run 1) measured 0.171s/epoch, 2.3-4.6x slower than every subsequent MPS run (compile-0 or compile-1) in the same session, each a fresh process. All later MPS runs (4 for compile-0, 4 for compile-1, alternating) clustered tightly (0.069-0.077 and 0.090-0.096 respectively). This looks like a one-time cost paid on the first-ever MPS/Metal invocation in the session (e.g. Metal shader compilation cache being cold on disk, not per-process) rather than genuine compile-vs-no-compile signal, since it appeared on a `--compile-model 0` run. The table above reports the steady-state average (excluding that one outlier); the raw run-by-run numbers are preserved in the session scratchpad driver output for reference. This is the same class of measurement caveat Task 1 flagged for GX10 (environment-specific first-run cost, not part of the compile question itself) — worth keeping in mind for any future single-shot MPS benchmark on this or other modules.

**Bottom line:** wiring `compile_model_if_enabled` into `image_classification` is correct and makes `--compile-model` functional for this script, matching the other reference scripts and Task 1's radar wiring. As with radar, the evidence here does not support turning `--compile-model` on by default for this small a model: it's a net loss on both CPU and MPS for `CNN_LENET5` at this input size/batch size, though the loss is proportionally smaller on MPS than on CPU (unlike radar, where both devices regressed by a similar percentage). A larger, more compute-bound image model (e.g. `CNN_IMG_MOBILENETV1_58K_NPU`/`CNN_IMG_MOBILENETV2_58K_NPU`) was not benchmarked here (the brief calls for the smallest/fastest model to keep iteration time reasonable) and might show a different tradeoff, since more per-op dispatch overhead in eager mode gives kernel fusion more to recoup.

**GX10 leg (controller-run):** same fixture/model/batch-size, 30 epochs, `~/jupyterlab/.venv` (torch 2.9.0+cu130), `NVIDIA GB10`. GX10's clone fast-forwarded cleanly to `bf261a7` this time (no reset needed, unlike Task 1's stale-clone situation). Same `cmsisdsp`/`torchaudio` `sys.modules` stubs as Task 1 (still unrelated to image_classification's own code path); no new missing packages this time — everything Task 1 installed already covered it.

| Config | Device | s/epoch | compile engaged? |
|---|---|---|---|
| No compile | CPU | 0.097 | n/a |
| No compile | CUDA | 0.074 | n/a |
| `--compile-model 1` | CPU | 0.198 | yes, `aot_eager`, no fallback |
| `--compile-model 1` | CUDA | 0.113 | attempted `inductor`, failed, fell back to eager (same Triton/gcc `cuda_utils.c` build failure as Task 1) |

**Finding: CUDA beats CPU on GX10 even without compile, but only by 1.3x here** (0.074 vs 0.097s/epoch) — a much smaller margin than radar's 4.8x. `CNN_LENET5`'s conv/pool ops give GX10's CPU (a many-core Grace ARM chip, not a phone-class CPU) enough to work with that the GPU's advantage narrows considerably for a model this tiny, compared to radar's pure-linear workload.

**Finding: `aot_eager` on CPU is a bigger loss on GX10 than on the Mac for this model** — 0.097 -> 0.198s/epoch is a ~104% regression on GX10's CPU, roughly consistent in direction and rough magnitude with the Mac's own CPU regression for image_classification (~124%) and both are much worse than either machine's CPU regression for radar. Small-CNN `aot_eager` compilation overhead appears to be a bigger relative cost than small-linear-net overhead was, on both machines.

**Finding: the same CUDA/Triton/gcc build failure from Task 1 reproduces identically here** — confirms it's an environment-level GX10 toolchain issue (Triton's nvidia backend `cuda_utils.c` failing to compile via `gcc`), not something specific to radar's model or code path. This will block any real `inductor`-backend measurement on GX10 across all modules until fixed there; worth investigating separately if CUDA compile behavior needs to be known for real, rather than re-discovering the same failure in Task 3's audio benchmark.

---

## Results (Task 3: audio_classification)

**Wiring:** `compile_model_if_enabled` is now imported and called in `audio_classification/train.py`'s `main()`, between `move_model_to_device` and `setup_distributed_model`, at the same indentation/position as radar's wiring (no extra `try:` nesting, unlike image_classification). Regression test `tests/test_audio_classification_compile_wired.py` passes; `--compile-model 0` (default) produces zero compile-related log lines and identical behavior to before this change; `--compile-model 1` produces `compile_model_if_enabled`'s own log line (`Compiling model with torch.compile (backend=aot_eager)`) on both CPU and MPS, with no fallback warnings on either device.

**X vs X_raw judgment call:** same resolution as Task 2, re-verified by reading `main()` directly rather than assumed. At the insertion point (right after `move_model_to_device`), `model` is still the plain model returned by `models.get_model(args.model, variables, num_classes, input_features=input_features, ...)` (or `torch.load` for `--load-saved-model`), built from `variables = dataset.X.shape[1]` / `input_features = tuple(dataset.X.shape[2:])` a few lines earlier (line 274-276) — confirmed by the existing `torchinfo.summary(model, summary_input_shape)` call at line 302-304 (gated on `args.generic_model or args.nas_enabled`), which also uses `dataset.X.shape[1:]` for this same pre-wrap model. The `nn_for_feature_extraction` / `X_raw` vs `X` branch the brief flagged only appears later (lines 322-331), where `model` gets wrapped in `NeuralNetworkWithPreprocess` (and, when `args.nn_for_feature_extraction` is set, a separately-trained `FEModelLinear` feature extractor consuming `dataset.X_raw`) — that wrapping happens *after* our insertion point, not before it, exactly as in Task 2. The `X_raw`-shaped input only matters again much later, at export time (line 424, `input_shape = (1,) + dataset.X_raw.shape[1:]` when `nn_for_feature_extraction` is set) — a separate, later insertion point this task doesn't touch. So regardless of `args.nn_for_feature_extraction`, the raw model being compiled at this point in `main()` always consumes `dataset.X.shape[1:]`-shaped input; the brief's `input_shape=(1,) + dataset.X.shape[1:]` is correct unconditionally, no branching needed.

**Fixture:** no existing audio fixture/driver in `tinyml-tinyverse/tests/` drives real audio data end-to-end (the closest prior art, `tests/test_train_best_epoch_bugs_vision_audio.py`, fully mocks `get_model`/`create_data_loaders`/etc. for both image and audio and never touches a real `GoogleSpeechCommandsDataset`). Built a small synthetic fixture instead: N class folders of 1-second, 16kHz mono WAV clips (`<root>/<class_name>/*.wav`, matching `GoogleSpeechCommandsDataset`'s fallback-discovery layout — `glob(<data_path>/*/*.wav)`, no `annotations/*_list.txt` needed since `--dataset` stays at its `folder` default), each class a distinct sine-tone frequency (200-3000 Hz, evenly spaced) plus Gaussian noise, generated with `soundfile` (confirmed `torchaudio.load` reads these back correctly with the installed `soundfile` backend — `torchaudio.list_audio_backends()` returns `['soundfile']` in this env). Model: `CNN_AUDIO_DSCNN`, the *only* model registered for `audio_classification` in `tinyml-modelzoo` (`tinyml_modelzoo/models/audio.py`) — a depthwise-separable-conv DSCNN (7 `Conv2d` layers, several depthwise, BatchNorm/ReLU/Dropout between them, ~23,171 parameters at default `filters=64`), a materially different op mix from radar's pure linear/BatchNorm stack and image's single small non-separable CNN. Its documented expected input is `(N, 1, 49, 10)`; the default audio CLI params (`sample_rate=16000`, `audio_duration_ms=1000` -> 16000 samples, `n_mfcc=10`, `frame_length_ms=30` -> `n_fft=480`, `frame_step_ms=20` -> `hop_length=320`) produce exactly 49 MFCC time frames x 10 coefficients with `center=False`, so no CLI overrides were needed to hit the model's expected shape. Driver script (`bench_audio.py` in the session scratchpad) builds argv for `train.get_args_parser().parse_args(argv)` and times `train.run(args)`, same pattern as `bench_radar.py`/`bench_image.py`.

**Methodology finding — `--sampling-rate` vs `--sample-rate` naming collision (pre-existing, not part of this task's fix):** while building the fixture, an initial run with `--sampling-rate 1.0` (the value Tasks 1/2 used for the base parser's generic, audio-irrelevant, `required=True` FFT `--sampling-rate` arg) crashed inside `torchaudio`'s MFCC transform with `RuntimeError: stft(...) : expected 0 < n_fft < 1, but got n_fft=0`. Root cause: `GoogleSpeechCommandsDataset.__init__` does `for key, value in kwargs.items(): setattr(self, key, ...)` over every CLI arg passed via `**vars(args)`, then reads `self.sampling_rate = int(getattr(self, "sampling_rate", 16000))` — but the *base* parser's generic `--sampling-rate` (dest `sampling_rate`, meant for radar/timeseries FFT preprocessing) collides with and silently overwrites the dataset's own `sampling_rate` attribute, while `audio_classification`'s own `--sample-rate` flag (dest `sample_rate`, no "ing") is never read by the dataset at all — it sets an unused attribute. With `--sampling-rate 1.0`, the dataset computed `n_fft = int(1 * 30 / 1000) = 0`, crashing MFCC extraction. Worked around by passing `--sampling-rate 16000` (the real audio sample rate) instead of `1.0` for this benchmark. This is a genuine, pre-existing naming/wiring bug in the audio dataset loader unrelated to `compile_model_if_enabled` and outside this task's file list (`audio_dataset.py`, not `audio_classification/train.py`) — not fixed here, flagged for separate follow-up.

**Manual E2E check (Step 5):** 3-class fixture (12 clips/class), 2 epochs, batch size 16, `--sampling-rate 16000`. `--device cpu --compile-model 0` (both pre- and post-code-change): completes cleanly, exports `model.onnx`, zero compile-related log lines. `--device cpu --compile-model 1`: completes cleanly, exports `model.onnx`, log shows `INFO: root.main: Compiling model with torch.compile (backend=aot_eager)` with no fallback warning. `--device mps --compile-model 0` and `--compile-model 1` (`PYTORCH_ENABLE_MPS_FALLBACK=1`): both complete cleanly and export `model.onnx`; compile-1 shows the same `aot_eager` compile line, no `compile_model_if_enabled` fallback warning, but does show a PyTorch `UserWarning` that `aten::native_dropout` isn't supported on MPS and falls back to CPU for that op — see the benchmark finding below.

**Benchmark (Step 6)** — 5-class fixture (20 clips/class = 100 clips), `CNN_AUDIO_DSCNN`, batch size 16, 30 epochs, `--device cpu` / `--device mps` (`PYTORCH_ENABLE_MPS_FALLBACK=1`), each config run 3-4 times to check stability (whole-`train.run()` timing, same methodology as Tasks 1-2, so includes one-time dataset-load/MFCC-extraction and ONNX-export overhead amortized over 30 epochs):

| Config | Device | s/epoch (avg) | compile engaged? |
|---|---|---|---|
| `--compile-model 0` (baseline) | CPU | 1.242 (n=3, range 1.222-1.254) | n/a (no compile lines in log) |
| `--compile-model 0` (baseline) | MPS | 0.110 (n=4, range 0.108-0.114) | n/a (no compile lines in log) |
| `--compile-model 1` | CPU | 1.186 (n=3, range 1.170-1.204) | yes, `aot_eager`, no fallback |
| `--compile-model 1` | MPS | 0.207 (n=4, range 0.201-0.211) | yes, `aot_eager`, no fallback (but see dropout finding below) |

**Finding: audio_classification matches neither radar's nor image's pattern — MPS massively outperforms CPU here, the opposite of both prior tasks, and `torch.compile` is a small net win on CPU but a large net loss on MPS.** Without compile, MPS is ~11.3x *faster* than CPU (0.110s vs 1.242s/epoch) — a complete inversion of radar (MPS 1.7x slower) and image (MPS ~2x slower). A quick 1-epoch-vs-30-epoch differential run confirms this isn't a fixed-cost artifact: isolating the one-time dataset-load overhead (~0.28s CPU, ~0.58s MPS) from steady-state per-epoch cost still gives CPU ≈1.21s/epoch vs MPS ≈0.09s/epoch, a ~13x native throughput gap. `CNN_AUDIO_DSCNN` is a real, moderately-sized conv-heavy model (23,171 params, 7 `Conv2d` layers incl. depthwise) operating on a 49x10 feature map at batch 16 — enough actual matmul/convolution work that Apple's GPU wins decisively, unlike radar's pure-linear workload or image's tiny few-hundred-parameter `CNN_LENET5`.

With compile enabled, CPU improves slightly (1.242 -> 1.186s/epoch, ~-4.5%) — the only device across all three tasks where `torch.compile` was a net *win*, however small — while MPS regresses sharply (0.110 -> 0.207s/epoch, ~+88%). This **narrows** the CPU/MPS gap the same direction as image_classification did (11.3x -> 5.7x) but starting from the opposite baseline (MPS ahead, not behind). The MPS regression has a concrete, observed mechanism rather than just inferred dynamo/guard overhead: the `--compile-model 1` MPS log (and only that log — the `--compile-model 0` MPS log has zero such warnings, despite the model using the same `Dropout(0.2)`/`Dropout(0.4)` layers in both cases) shows `UserWarning: The operator 'aten::native_dropout' is not currently supported on the MPS backend and will fall back to run on the CPU`. This means compiling routes dropout through a lower-level decomposition (`aten::native_dropout`, likely via dynamo/AOTAutograd's graph capture) that lacks an MPS kernel, forcing a real CPU round-trip on every forward pass in the compiled path — whereas eager mode's `F.dropout` apparently dispatches through a path with native MPS support. A finer-grained 1-epoch-vs-30-epoch differential on the compiled MPS runs suggests the one-time `torch.compile` warmup cost (~1.4s) is also non-trivial relative to the 30-epoch total, so the naive 30-epoch-average ~88% regression somewhat overstates the true steady-state per-epoch cost (~56% by that isolation) — still a clear, large loss either way, just with the same one-time-warmup-dilution caveat Task 2 flagged for its MPS cold-start artifact.

**Bottom line:** wiring `compile_model_if_enabled` into `audio_classification` is correct and makes `--compile-model` functional for this script, matching all six other reference scripts. Unlike radar and image_classification, the evidence here does *not* uniformly argue against compile: CPU sees a small, consistent win, and the MPS regression has an identified, addressable-in-principle cause (a dropout decomposition lacking an MPS kernel under `torch.compile`) rather than being an unavoidable property of small-model dynamo overhead. Still, the MPS loss is large enough (and CPU's win small enough) that there's no case for flipping the opt-in default here either — `--compile-model` should stay opt-in, as it does today, but audio_classification is the first of the three tasks where "does compile help" doesn't have a uniformly negative answer, and the dropout/MPS-kernel angle is a concrete, non-speculative lead if `torch.compile` + MPS support for this workload is ever revisited.

**GX10 leg (controller-run):** same fixture/model/batch-size, 30 epochs, `~/jupyterlab/.venv` (torch 2.9.0+cu130), `NVIDIA GB10`. Clone fast-forwarded cleanly to `faad516`. Environment needed real (not stubbed) `torchaudio` this time, since `GoogleSpeechCommandsDataset` genuinely calls `torchaudio.load`/`MFCC`/`resample` (unlike radar/image, where `cmsisdsp`/`torchaudio` were only pulled in as unused eager imports and safely stubbed via `sys.modules`). `pip install torchaudio` grabbed 2.11.0 by default — ABI-incompatible with torch 2.9.0 (`undefined symbol: torch_library_impl` on import). Pinning `torchaudio==2.9.0` matched cleanly, but its default audio backend (`torchcodec`) requires system FFmpeg, which isn't installed on GX10 and isn't worth adding just for this benchmark. Worked around by monkey-patching `torchaudio.load` in the GX10 driver script only (not the repo) to read via the already-installed `soundfile` instead — this only changes how the benchmark's WAV fixture gets loaded into a tensor, not any training/compile code path, so it doesn't affect what's being measured.

| Config | Device | s/epoch | compile engaged? |
|---|---|---|---|
| No compile | CPU | 0.174 | n/a |
| No compile | CUDA | 0.079 | n/a |
| `--compile-model 1` | CPU | 0.305 | yes, `aot_eager`, no fallback |
| `--compile-model 1` | CUDA | 0.141 | attempted `inductor`, failed (same Triton/gcc `cuda_utils.c` build failure as Tasks 1 and 2), fell back to eager |

**Finding: CUDA beats CPU 2.2x on GX10 without compile** (0.079 vs 0.174s/epoch) — between radar's 4.8x and image's 1.3x, consistent with `CNN_AUDIO_DSCNN`'s real conv workload giving the GPU a clear but not overwhelming edge over GX10's capable CPU.

**Finding: unlike the Mac, GX10's CPU `aot_eager` backend is a clear net LOSS for audio (+75%, 0.174 -> 0.305s/epoch) — the opposite sign from the Mac's small CPU win (~-4.5%).** This is the first case across all three tasks and both machines where the same model/backend combination flips sign between machines. Since GX10's CUDA compile attempt never actually ran (Triton/gcc failure, same as every other module on this box), there's no GX10 MPS-dropout-style mechanism to check on the GPU side — CUDA does have a native dropout kernel, so the MPS-specific finding from the Mac leg doesn't apply here regardless.

**Finding: the Triton/gcc `inductor` build failure reproduces identically for the third module in a row** — now confirmed systemic across all of radar, image_classification, and audio_classification on this GX10 environment, not tied to any particular model or code path.

**Net across all three tasks and two machines: no single "compile helps" or "compile hurts" story holds everywhere.** Radar and image lose on both Mac devices; audio wins narrowly on Mac CPU and loses hard on Mac MPS (with a diagnosed cause); GX10 CPU flips sign between image/radar-style modules and audio; GX10 CUDA's real compile behavior remains entirely unmeasured across all three modules due to one shared environment issue. `--compile-model` staying opt-in (its current default) is the only conclusion that holds up across all of this data.

**GX10 leg:** not run for this task — the Task 3 brief's Step 6 scopes the benchmark to "same methodology as Task 2 Step 6" (the Mac CPU/MPS 4-way comparison); the GX10 legs recorded under Tasks 1 and 2 were performed by a separate controller-run process outside this task's instructions, not part of this subagent's assigned steps.
