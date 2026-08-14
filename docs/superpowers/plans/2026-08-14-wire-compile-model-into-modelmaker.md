# Wire compile_model into modelmaker for Radar, Vision, and Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--compile-model` reachable from the supported `tinyml-modelmaker` production path for `radar`, `vision` (image), and `audio` — closing the gap an independent peer review found in the `compile-hardening-radar-image-audio` plan: `compile_model_if_enabled` was wired into all three `tinyml-tinyverse` scripts, but none of the three modules' `tinyml-modelmaker` orchestration layers ever pass `--compile-model`, so the flag is permanently `0` (its default) for anyone using the actual product, not just direct script invocation.

**Architecture:** `timeseries` already has this wiring end to end and is the template: a `compile_model=0` field in its `params.py` training dict, an `apply_hardware_defaults(params, user_training_keys)` call at the end of `init_params()` (which auto-flips `compile_model` to `1` when CUDA is available and the user hasn't explicitly set it), and a `'--compile-model', f'{getattr(self.params.training, "compile_model", 0)}'` entry in its argv builder. `apply_hardware_defaults` itself (`tinyml-modelmaker/tinyml_modelmaker/utils/hardware_defaults.py`) already guards every field access with `hasattr`, and its own docstring says it was built for exactly this rollout ("hasattr guards keep this safe for params that don't carry these fields yet (vision, audio — Phase 2)") — so `apply_hardware_defaults` itself needs no changes, only the three modules' own params.py/argv-builder files.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, pytest

## Global Constraints

- Python `==3.10.*`
- No new dependencies
- Do not modify `apply_hardware_defaults` itself (`tinyml-modelmaker/tinyml_modelmaker/utils/hardware_defaults.py`) — it already supports this rollout via its `hasattr` guards
- Do not modify `timeseries`'s existing wiring (reference only)
- `compile_model` must default to `0` in each module's params (matching timeseries) — `apply_hardware_defaults` is what conditionally raises it to `1`, not the static default
- Each task's fix must be verified against a real training run through the `tinyml-modelmaker` layer (not just a unit test of the params/argv construction), since this is exactly the integration layer that direct-script-only testing has already been shown (twice, in prior plans) to miss real bugs in

## Context: Why This Is Needed

`docs/superpowers/plans/2026-08-13-compile-hardening-radar-image-audio.md` wired `compile_model_if_enabled` into `radar_classification`, `image_classification`, and `audio_classification`'s `main()` functions in `tinyml-tinyverse`, and its own stated Goal was "so all seven reference training scripts get the same hardware-acceleration path." An independent peer code review found this Goal was only half-achieved: `grep -rn compile_model tinyml-modelmaker/tinyml_modelmaker/ai_modules/{radar,vision,audio}/` returns nothing, versus `timeseries/params.py:160,232` and `timeseries_base.py:765`, which have the full three-piece wiring. Since `tinyml-modelmaker` is the actual product surface (the `tinyml-tinyverse` reference scripts are the layer it drives, not something end users invoke directly), the compile wiring from the prior plan currently does nothing for anyone using radar/vision/audio through the product.

This plan closes that gap using the exact pattern `timeseries` and `apply_hardware_defaults` already establish — confirmed by direct inspection of all four modules' current source before writing this plan (not assumed by analogy).

---

## File Map

| Action | Path | Responsibility |
|--------|------|-----------------|
| Modify | `tinyml-modelmaker/tinyml_modelmaker/ai_modules/radar/params.py` | Add `compile_model=0` field + `apply_hardware_defaults` call |
| Modify | `tinyml-modelmaker/tinyml_modelmaker/ai_modules/radar/training/tinyml_tinyverse/radar_base.py` | Add `--compile-model` to train argv |
| Modify | `tinyml-modelmaker/tinyml_modelmaker/ai_modules/vision/params.py` | Same as radar |
| Modify | `tinyml-modelmaker/tinyml_modelmaker/ai_modules/vision/training/tinyml_tinyverse/image_base.py` | Same as radar |
| Modify | `tinyml-modelmaker/tinyml_modelmaker/ai_modules/audio/params.py` | Same as radar |
| Modify | `tinyml-modelmaker/tinyml_modelmaker/ai_modules/audio/training/tinyml_tinyverse/audio_base.py` | Same as radar |

---

## Task 1: Wire compile_model into radar's modelmaker layer

**Files:**
- Modify: `tinyml-modelmaker/tinyml_modelmaker/ai_modules/radar/params.py` (training dict at line 80; end of `init_params` at lines 198-199)
- Modify: `tinyml-modelmaker/tinyml_modelmaker/ai_modules/radar/training/tinyml_tinyverse/radar_base.py` (`_build_common_train_argv`, line 317)

**Interfaces:**
- Consumes: `tinyml_modelmaker.utils.hardware_defaults.apply_hardware_defaults` (existing, unmodified, already imported this way in `timeseries/params.py:38`)

- [ ] **Step 1: Write the failing test**

```python
"""Regression test: radar's modelmaker params must carry a compile_model
field, and apply_hardware_defaults must be invoked so it can auto-enable
compile on CUDA -- matching the pattern timeseries already has. Without
this, --compile-model is unreachable from the modelmaker (product) path
even though tinyml-tinyverse's radar_classification.train.main() already
supports it."""
from tinyml_modelmaker.ai_modules.radar.params import init_params


def test_init_params_carries_compile_model_field():
    params = init_params()
    assert hasattr(params.training, "compile_model"), (
        "radar's params.training has no compile_model field -- "
        "apply_hardware_defaults can't act on it (it's hasattr-guarded), "
        "and the field never reaches the --compile-model argv flag."
    )
    assert params.training.compile_model == 0, (
        "compile_model must default to 0 (matching timeseries) -- "
        "apply_hardware_defaults is what conditionally raises it, not the static default."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tinyml-modelmaker && python -m pytest tests/test_radar_compile_model_param.py -v`
Expected: FAIL — `AttributeError` or `hasattr` returns `False`, `compile_model` not present

- [ ] **Step 3: Add the field to radar/params.py's training dict**

In `tinyml-modelmaker/tinyml_modelmaker/ai_modules/radar/params.py`, in the `training=dict(...)` block, add `compile_model=0,` immediately before the `training_device='cuda',  # 'cpu', 'cuda'` line:

```python
            momentum=0,
            compile_model=0,    # 1 to enable torch.compile (inductor on CUDA, aot_eager on MPS)
            training_device='cuda',  # 'cpu', 'cuda'
```

- [ ] **Step 4: Add the apply_hardware_defaults call**

Add the import near the top of the file (matching `timeseries/params.py:38`'s exact form):
```python
from ...utils.hardware_defaults import apply_hardware_defaults
```

Change the end of `init_params` from:
```python
    params = utils.ConfigDict(default_params, *args, **kwargs)
    return params
```
to:
```python
    user_training_keys = set(args[0].get('training', {}).keys()) \
        if args and isinstance(args[0], dict) else set()
    params = utils.ConfigDict(default_params, *args, **kwargs)
    apply_hardware_defaults(params, user_training_keys)
    return params
```
(matching `timeseries/params.py:227-232` exactly, including the comment above `user_training_keys` in that file explaining why the `isinstance` check exists — copy it verbatim for consistency.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd tinyml-modelmaker && python -m pytest tests/test_radar_compile_model_param.py -v`
Expected: PASS

- [ ] **Step 6: Wire --compile-model into radar_base.py's train argv**

In `tinyml-modelmaker/tinyml_modelmaker/ai_modules/radar/training/tinyml_tinyverse/radar_base.py`'s `_build_common_train_argv`, change:
```python
            '--distributed', f'{distributed}',
            '--device', f'{device}',

            '--generic-model', f'{self.params.common.generic_model}',
```
to:
```python
            '--distributed', f'{distributed}',
            '--device', f'{device}',
            '--compile-model', f'{getattr(self.params.training, "compile_model", 0)}',

            '--generic-model', f'{self.params.common.generic_model}',
```

- [ ] **Step 7: Write a test confirming the argv actually carries it**

```python
"""Regression test: radar_base.py's train argv must include --compile-model,
sourced from params.training.compile_model -- otherwise the field added in
Step 3 has nowhere to go and remains dead."""
from tinyml_modelmaker.ai_modules.radar.params import init_params
from tinyml_modelmaker.ai_modules.radar.training.tinyml_tinyverse.radar_base import BaseRadarModelTraining


def test_build_common_train_argv_includes_compile_model():
    params = init_params()
    params.training.compile_model = 1

    class _Dummy(BaseRadarModelTraining):
        train_module = None
        test_module = None

    instance = object.__new__(_Dummy)
    instance.params = params

    argv = instance._build_common_train_argv(device="cpu", distributed=0)
    assert "--compile-model" in argv, "argv builder never emits --compile-model"
    idx = argv.index("--compile-model")
    assert argv[idx + 1] == "1", (
        f"--compile-model should carry params.training.compile_model's value (1), got {argv[idx + 1]!r}"
    )
```

Run: `cd tinyml-modelmaker && python -m pytest tests/test_radar_compile_model_param.py -v` (add this test to the same file)
Expected: PASS after Step 6's change, would FAIL before it

- [ ] **Step 8: Manual end-to-end verification through the modelmaker layer**

This is the layer prior plans' direct-script-only testing has already missed real bugs in twice — verify for real, not just via unit tests. Using the synthetic radar fixture (`make_radar_fixture.py` from the session scratchpad, or regenerate per `docs/superpowers/plans/2026-08-13-radar-training-entrypoint-fix.md`'s Task 1), drive radar training through `BaseRadarModelTraining.run()` (not directly through `tinyml_tinyverse.references.radar_classification.train`) with `params.training.compile_model = 1` set explicitly, `--device cpu`, a couple of epochs. Confirm the run log shows `compile_model_if_enabled`'s own `Compiling model with torch.compile` INFO line — proving the flag's value actually reaches the trainer through the full modelmaker → tinyverse argv-passing chain, not just that the argv list contains the right string.

- [ ] **Step 9: Commit**

```bash
cd tinyml-modelmaker
git add tinyml_modelmaker/ai_modules/radar/params.py tinyml_modelmaker/ai_modules/radar/training/tinyml_tinyverse/radar_base.py tests/test_radar_compile_model_param.py
git commit -m "feat: wire compile_model into radar's modelmaker params/argv/hardware-defaults"
```

---

## Task 2: Wire compile_model into vision's (image_classification's) modelmaker layer

**Files:**
- Modify: `tinyml-modelmaker/tinyml_modelmaker/ai_modules/vision/params.py` (training dict at line 80; end of `init_params` at lines 226-227)
- Modify: `tinyml-modelmaker/tinyml_modelmaker/ai_modules/vision/training/tinyml_tinyverse/image_base.py` (`_build_common_train_argv`, line 337)

**Interfaces:**
- Same as Task 1, applied to the vision module. Independent of Task 1 — same pattern, different files.

- [ ] **Step 1: Write the failing test**

```python
"""Regression test: vision's modelmaker params must carry a compile_model
field, and apply_hardware_defaults must be invoked so it can auto-enable
compile on CUDA -- matching the pattern timeseries already has."""
from tinyml_modelmaker.ai_modules.vision.params import init_params


def test_init_params_carries_compile_model_field():
    params = init_params()
    assert hasattr(params.training, "compile_model"), (
        "vision's params.training has no compile_model field -- "
        "apply_hardware_defaults can't act on it (it's hasattr-guarded), "
        "and the field never reaches the --compile-model argv flag."
    )
    assert params.training.compile_model == 0, (
        "compile_model must default to 0 (matching timeseries) -- "
        "apply_hardware_defaults is what conditionally raises it, not the static default."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tinyml-modelmaker && python -m pytest tests/test_vision_compile_model_param.py -v`
Expected: FAIL — `compile_model` not present on `params.training`

- [ ] **Step 3: Add `compile_model=0,` to vision/params.py's training dict**, immediately before `training_device=constants.TRAINING_DEVICE_CUDA,`.

- [ ] **Step 4: Add the `apply_hardware_defaults` import and call**, same exact pattern as Task 1 Step 4, applied to `vision/params.py`'s `init_params`.

- [ ] **Step 5: Run test to verify it passes**

- [ ] **Step 6: Wire `--compile-model` into image_base.py's train argv.** In `_build_common_train_argv` (line 337 area), insert after `'--device', f'{device}',` — same pattern as Task 1 Step 6, adapted to this file's exact surrounding lines (confirm the exact `--generic-model`/next-line context before editing, since `image_base.py` also has a `--sampling-rate` line here per the file map already inspected — insert `--compile-model` right after `--device`, before those).

- [ ] **Step 7: Write the argv-carries-it test**, mirroring Task 1 Step 7, targeting `ModelTraining`/`BaseModelTraining` in `image_base.py` (check the exact base class name in this file — it likely differs from radar's `BaseRadarModelTraining`, confirm before writing).

- [ ] **Step 8: Manual end-to-end verification.** Reuse the image fixture pattern from `docs/superpowers/plans/2026-08-13-compile-hardening-radar-image-audio.md` Task 2 (`make_image_fixture.py`), drive training through vision's modelmaker `run()` (not directly through `tinyml_tinyverse.references.image_classification.train`) with `compile_model = 1` set, confirm the compile INFO log line appears.

- [ ] **Step 9: Commit**

```bash
cd tinyml-modelmaker
git add tinyml_modelmaker/ai_modules/vision/params.py tinyml_modelmaker/ai_modules/vision/training/tinyml_tinyverse/image_base.py tests/test_vision_compile_model_param.py
git commit -m "feat: wire compile_model into vision's modelmaker params/argv/hardware-defaults"
```

---

## Task 3: Wire compile_model into audio's modelmaker layer

**Files:**
- Modify: `tinyml-modelmaker/tinyml_modelmaker/ai_modules/audio/params.py` (training dict at line 79; end of `init_params` at lines 213-214)
- Modify: `tinyml-modelmaker/tinyml_modelmaker/ai_modules/audio/training/tinyml_tinyverse/audio_base.py` (`_build_common_train_argv`, line 317)

**Interfaces:**
- Same as Tasks 1/2, applied to the audio module. Independent of both.

- [ ] **Step 1: Write the failing test**

```python
"""Regression test: audio's modelmaker params must carry a compile_model
field, and apply_hardware_defaults must be invoked so it can auto-enable
compile on CUDA -- matching the pattern timeseries already has."""
from tinyml_modelmaker.ai_modules.audio.params import init_params


def test_init_params_carries_compile_model_field():
    params = init_params()
    assert hasattr(params.training, "compile_model"), (
        "audio's params.training has no compile_model field -- "
        "apply_hardware_defaults can't act on it (it's hasattr-guarded), "
        "and the field never reaches the --compile-model argv flag."
    )
    assert params.training.compile_model == 0, (
        "compile_model must default to 0 (matching timeseries) -- "
        "apply_hardware_defaults is what conditionally raises it, not the static default."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tinyml-modelmaker && python -m pytest tests/test_audio_compile_model_param.py -v`
Expected: FAIL — `compile_model` not present on `params.training`

- [ ] **Step 3: Add `compile_model=0,` to audio/params.py's training dict**, immediately before `training_device='cuda',  # 'cpu', 'cuda'`.

- [ ] **Step 4: Add the `apply_hardware_defaults` import and call**, same pattern as Tasks 1/2.

- [ ] **Step 5: Run test to verify it passes**

- [ ] **Step 6: Wire `--compile-model` into audio_base.py's train argv.** In `_build_common_train_argv` (line 317 area), insert after `'--device', f'{device}',`, matching this file's exact surrounding structure (confirmed identical to radar's at this point per the earlier file map inspection — `'--generic-model'` follows).

- [ ] **Step 7: Write the argv-carries-it test**, mirroring Task 1 Step 7.

- [ ] **Step 8: Manual end-to-end verification.** Reuse the audio fixture pattern from `docs/superpowers/plans/2026-08-13-compile-hardening-radar-image-audio.md` Task 3 (`make_audio_fixture.py`), drive training through audio's modelmaker `run()` with `compile_model = 1` set, confirm the compile INFO log line appears. Watch for the `--sampling-rate` requirement (`required=True` on the base parser) — audio_base.py already supplies it correctly (per the separately-fixed `--sample-rate` dead-flag plan), so this should just work, but confirm rather than assume.

- [ ] **Step 9: Commit**

```bash
cd tinyml-modelmaker
git add tinyml_modelmaker/ai_modules/audio/params.py tinyml_modelmaker/ai_modules/audio/training/tinyml_tinyverse/audio_base.py tests/test_audio_compile_model_param.py
git commit -m "feat: wire compile_model into audio's modelmaker params/argv/hardware-defaults"
```

---

## Note for the final whole-plan review

Once all three tasks land, re-verify `apply_hardware_defaults`'s CUDA-only auto-enable behavior doesn't regress anything on GX10: with all three modules now carrying `compile_model` fields, a modelmaker-driven run on GX10 (where `torch.cuda.is_available()` is `True`) will auto-set `compile_model=1` unless the user explicitly configured it — meaning `inductor` will now auto-engage for radar/vision/audio through the product path on any CUDA machine. Given this plan's own sibling plan found `inductor` to be a large *regression* at short (30-epoch) run lengths for these three models, this is worth flagging explicitly in the whole-plan review as a real behavior change on CUDA hardware, not just a "flag now works" story — cross-reference `docs/superpowers/plans/2026-08-13-compile-hardening-radar-image-audio.md`'s Addendum section for the actual numbers.

**Decision (2026-08-14, repo owner, following the whole-plan review):** keep auto-enable, matching `timeseries`'s existing behavior — no change to `apply_hardware_defaults` or its scope. Rationale: the sibling plan's regressions were measured on very short (30-epoch) synthetic benchmarks where one-time `torch.compile`/`inductor` warmup dominates the total; real training runs are almost certainly long enough to amortize that fixed cost and come out ahead, which is the same reasoning `apply_hardware_defaults` already applies to `timeseries`. Radar/vision/audio now behave consistently with `timeseries` rather than being a special case. If this assumption turns out to be wrong for real workloads (not just this session's tiny synthetic fixtures), revisit by either excluding these three modules from `apply_hardware_defaults`'s auto-enable or exposing `compile_model` in the params `descriptions` block so users can see and override it from the GUI (currently YAML-only, noted as a gap by the whole-plan review).
