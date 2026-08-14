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
